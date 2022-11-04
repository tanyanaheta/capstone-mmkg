import argparse
import os

import dgl
import dgl.function as fn
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.datamodules.negative_sampler import NegativeSampler
from src.model.SAGE import SAGE


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
        n_layers=3,
        activation=F.relu,
        dropout=0,
        sage_conv_method="mean",
        lr=0.0005,
        batch_size=1024,
    ):
        super().__init__()
        self.module = SAGE(
            in_dim, h_dim, n_layers, activation, dropout, sage_conv_method
        )
        self.lr = lr
        self.predictor = ScorePredictor()
        self.batch_size = batch_size
        self.save_hyperparameters()

        self.train_loss = MeanMetric()
        self.val_positive_distance = MeanMetric()
        self.val_negative_distance = MeanMetric()

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
        self.train_loss(loss)
        self.log(
            "train_loss",
            self.train_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, pos_graph, neg_graph, blocks = batch
        x = blocks[0].srcdata["feat"]
        logits = self.module(blocks, x)
        pos_score = self.predictor(pos_graph, logits)
        neg_score = self.predictor(neg_graph, logits)

        self.val_positive_distance(pos_score)
        self.val_negative_distance(neg_score)

        self.log(
            "mean_val_posititve_distance",
            self.val_positive_distance,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "mean_val_negative_distance",
            self.val_negative_distance,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class DataModule(LightningDataModule):
    def __init__(
        self,
        csv_dataset_root,
        data_cpu=False,
        fan_out=[10, 25],
        device="cpu",
        batch_size=1024,
        num_workers=4,
        force_reload=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        dataset = dgl.data.CSVDataset("/zillow_graph_csv", force_reload=force_reload)
        g = dataset[0]
        g, reverse_eids = to_bidirected_with_reverse_mapping(g)
        # g = g.formats(["csc"])
        g = g.to(device)
        reverse_eids = reverse_eids.to(device)
        # seed_edges = torch.arange(g.num_edges()).to(device)

        train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0].to(device)
        val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0].to(device)
        test_nid = torch.nonzero(
            ~(g.ndata["train_mask"] | g.ndata["val_mask"]), as_tuple=True
        )[0].to(device)

        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(_) for _ in fan_out], prefetch_node_feats=["feat"]
        )

        self.g = g
        self.train_nid, self.val_nid, self.test_nid = train_nid, val_nid, test_nid
        self.sampler = sampler
        self.device = device
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
            negative_sampler=NegativeSampler(self.g, 1)
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


@hydra.main(config_name="config", config_path="conf", version_base=None)
def train(cfg):
    datamodule = DataModule(
        cfg.data.zillow_root, device=device, batch_size=cfg.training.batch_size
    )
    model = SAGELightning(
        datamodule.in_dim,
        cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        batch_size=cfg.training.batch_size,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="mean_val_negative_distance", save_top_k=1, mode="max"
    )
    trainer = Trainer(accelerator="gpu", max_epochs=10, callbacks=[checkpoint_callback])

    trainer.fit(model, datamodule=datamodule)


@hydra.main(config_name="config", config_path="conf", version_base=None)
def evaluate(cfg):
    datamodule = DataModule(
        cfg.data.zillow_root, device=device, batch_size=cfg.training.batch_size
    )
    model = SAGELightning(
        datamodule.in_dim,
        h_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        batch_size=cfg.training.batch_size,
    )

    trainer = Trainer(accelerator="gpu")

    dataloader = datamodule.val_dataloader()

    trainer.test(model, dataloaders=dataloader)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    train()
    print("Done")
