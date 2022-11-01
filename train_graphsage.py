import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from torchmetrics import RetrievalMRR, RetrievalMAP
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from src.model import SAGE
from src.datamodules.negative_sampler import NegativeSampler
import dgl.function as fn
import dgl
from sklearn.metrics import roc_auc_score
from torchmetrics import AUROC


def to_bidirected_with_reverse_mapping(g):
    """Makes a graph bidirectional, and returns a mapping array ``mapping`` where ``mapping[i]``
    is the reverse edge of edge ID ``i``. Does not work with graphs that have self-loops.
    """
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g), return_counts="count", writeback_mapping=True
    )
    c = g_simple.edata["count"]
    num_edges = g.num_edges()
    mapping_offset = torch.zeros(g_simple.num_edges() + 1, dtype=g_simple.idtype)
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = torch.where(
        idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges
    )
    reverse_mapping = mapping[reverse_idx]
    # sanity check
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert torch.equal(src1, dst2)
    assert torch.equal(src2, dst1)
    return g_simple, reverse_mapping


# class CrossEntropyLoss(nn.Module):
#     def forward(self, block_outputs, pos_graph, neg_graph):
#         with pos_graph.local_scope():
#             pos_graph.ndata["h"] = block_outputs
#             pos_graph.apply_edges(fn.u_dot_v("h", "h", "score"))
#             pos_score = pos_graph.edata["score"]
#         with neg_graph.local_scope():
#             neg_graph.ndata["h"] = block_outputs
#             neg_graph.apply_edges(fn.u_dot_v("h", "h", "score"))
#             neg_score = neg_graph.edata["score"]

#         score = torch.cat([pos_score, neg_score])
#         label = torch.cat(
#             [torch.ones_like(pos_score), torch.zeros_like(neg_score)]
#         ).long()
#         loss = F.binary_cross_entropy_with_logits(score, label.float())
#         return loss


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata["h"] = x
            edge_subgraph.apply_edges(fn.u_dot_v("h", "h", "score"))
            return edge_subgraph.edata["score"]


class SAGELightning(LightningModule):
    def __init__(
        self,
        in_dim,
        h_dim,
        n_layers,
        activation=F.relu,
        dropout=0,
        sage_conv_method="mean",
        lr=0.0005,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.module = SAGE(
            in_dim, h_dim, n_layers, activation, dropout, sage_conv_method
        )
        self.lr = lr
        self.predictor = ScorePredictor()

    def forward(self, graph, blocks, x):
        self.module(graph, blocks, x)

    def training_step(self, batch, batch_idx):
        input_nodes, pos_graph, neg_graph, blocks = batch
        x = blocks[0].srcdata["feat"]
        logits = self.module(blocks, x)
        pos_score = self.predictor(pos_graph, logits)
        neg_score = self.predictor(neg_graph, logits)

        score = torch.cat([pos_score, neg_score])
        pos_label = torch.ones_like(pos_score)
        neg_label = torch.zeros_like(neg_score)
        labels = torch.cat([pos_label, neg_label])
        loss = F.binary_cross_entropy_with_logits(score, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, pos_graph, neg_graph, blocks = batch
        x = blocks[0].srcdata["feat"]
        logits = self.module(blocks, x)
        pos_score = self.predictor(pos_graph, logits)
        neg_score = self.predictor(neg_graph, logits)

        pos_score, neg_score = self.module(pos_graph, neg_graph, blocks, x)
        self.log(
            "mean_val_pos_score",
            pos_score.mean(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "mean_val_neg_score",
            neg_score.mean(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name,
        data_cpu=False,
        fan_out=[10, 25],
        device="cpu",
        batch_size=1024,
        num_workers=4,
        force_reload=False,
    ):
        super().__init__()
        if dataset_name == "zillow":
            dataset = dgl.data.CSVDataset(
                "./data/zillow_graph/zillow_graph_dgl", force_reload=force_reload
            )
            g = dataset[0]
            g = g.to(device)
            g, reverse_eids = to_bidirected_with_reverse_mapping(g)
            reverse_eids = reverse_eids.to(device)
            seed_edges = torch.arange(g.num_edges()).to(device)
        else:
            pass

        train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
        val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
        test_nid = torch.nonzero(
            ~(g.ndata["train_mask"] | g.ndata["val_mask"]), as_tuple=True
        )[0]

        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(_) for _ in fan_out], prefetch_node_feats=["feat"]
        )

        dataloader_device = torch.device("cpu")
        if not data_cpu:
            train_nid = train_nid.to(device)
            val_nid = val_nid.to(device)
            test_nid = test_nid.to(device)
            g = g.formats(["csc"])
            g = g.to(device)
            dataloader_device = device

        self.g = g
        self.train_nid, self.val_nid, self.test_nid = train_nid, val_nid, test_nid
        self.sampler = sampler
        self.device = dataloader_device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_dim = g.ndata["feat"].shape[1]
        self.reverse_eids = reverse_eids

    def train_dataloader(self):
        sampler = dgl.dataloading.as_edge_prediction_sampler(
            self.sampler,
            exclude="reverse_id",
            reverse_eids=self.reverse_eids,
            negative_sampler=NegativeSampler(self.g, 5)
            # negative_sampler=dgl.dataloading.negative_sampler.PerSourceUniform(5),
        )

        return dgl.dataloading.DataLoader(
            self.g,
            self.train_nid,
            sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            # num_workers=self.num_workers,
        )

    def val_dataloader(self):
        sampler = dgl.dataloading.as_edge_prediction_sampler(
            self.sampler,
            exclude="reverse_id",
            reverse_eids=self.reverse_eids,
            negative_sampler=NegativeSampler(self.g, 5)
            # negative_sampler=dgl.dataloading.negative_sampler.PerSourceUniform(5),
        )

        return dgl.dataloading.DataLoader(
            self.g,
            self.val_nid,
            sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            # num_workers=self.num_workers,
        )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        mode = "cpu"
    print(f"Training in {mode} mode.")

    print("Loading data")
    datamodule = DataModule("zillow", device=mode)

    model = SAGELightning(datamodule.in_dim, 256, 3)

    checkpoint_callback = ModelCheckpoint(monitor="mean_val_pos_score", save_top_k=1)
    trainer = Trainer(gpus=[0], max_epochs=2, callbacks=[checkpoint_callback])

    trainer.fit(model, datamodule=datamodule)
