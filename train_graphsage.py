import argparse
import os
import json

import dgl
import dgl.function as fn
import hydra
import numpy as np
import pyrootutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

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
            edge_subgraph.ndata['h_norm'] = F.normalize(x, p=2, dim=-1)
            edge_subgraph.apply_edges(fn.u_dot_v("h_norm", "h_norm", "score"))
            return edge_subgraph.edata["score"]


class SAGELightning(LightningModule):
    def __init__(
        self,
        in_dim,
        h_dim,
        n_layers=3,
        activation=F.relu,
        dropout=0.7,
        sage_conv_method="mean",
        lr=0.005,
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
        self.mean_val_positive_score = MeanMetric()
    
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

        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, pos_graph, blocks = batch
        x = blocks[0].srcdata["feat"]
        logits = self.module(blocks, x)
        pos_score = self.predictor(pos_graph, logits)
        pos_label = torch.ones_like(pos_score)
        self.mean_val_positive_score(pos_score)

        self.log(
            "mean_val_positive_score",
            self.mean_val_positive_score,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class NegativeSamplerTest(object):
    def __init__(self, g, k, max_img_id, keyword_as_src, neg_share=False):
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k
        self.neg_share = neg_share
        self.max_img_id = max_img_id
        self.keyword_as_src = keyword_as_src

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        if self.keyword_as_src == False:
            img_node_mask = src <= self.max_img_id
            src = src[img_node_mask]
        n = len(src)

        if self.neg_share and n % self.k == 0:
            dst = self.weights.multinomial(n, replacement=True)
            dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
        else:
            dst = self.weights.multinomial(n * self.k, replacement=True)
            
        src = src.repeat_interleave(self.k)
        return src, dst

class DataModule(LightningDataModule):
    def __init__(
        self,
        csv_dataset_root,
        modal_node_ids_file,
        keyword_as_src=False,
        data_cpu=False,
        fan_out=[3],
        device="cpu",
        batch_size=1024,
        num_workers=4,
        force_reload=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        dataset = dgl.data.CSVDataset(csv_dataset_root, force_reload=force_reload)
        g = dataset[0]
        g_bid, reverse_eids = to_bidirected_with_reverse_mapping(g)
        g_bid = g_bid.to(device)
        g = g.to(device)
        reverse_eids = reverse_eids.to(device)

        max_img_id = max(json.load(open(modal_node_ids_file, 'r'))['images'])

        train_nid = torch.nonzero(g_bid.ndata["train_mask"], as_tuple=True)[0].to(device)
        val_nid = torch.nonzero(g_bid.ndata["val_mask"], as_tuple=True)[0].to(device)
        test_nid = torch.nonzero(g_bid.ndata["test_mask"], as_tuple=True)[0].to(device)

        sampler = dgl.dataloading.NeighborSampler(fan_out)

        self.g = g
        self.g_bid = g_bid
        self.train_nid = train_nid
        self.val_nid = val_nid #torch.cat((val_nid, test_nid))
        self.sampler = sampler
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_dim = g_bid.ndata["feat"].shape[1]
        self.reverse_eids = reverse_eids
        self.max_img_id = max_img_id
        self.keyword_as_src = keyword_as_src


    def train_dataloader(self):
        edge_sampler = dgl.dataloading.as_edge_prediction_sampler(
            self.sampler,
            exclude='reverse_id',
            reverse_eids=self.reverse_eids,
            negative_sampler=NegativeSamplerTest(self.g, 1, self.max_img_id, self.keyword_as_src)
        )

        train_subgraph = self.g_bid.subgraph(self.train_nid)
        train_u, train_v = train_subgraph.edges()
        train_eids = train_subgraph.edata['_ID'][train_subgraph.edge_ids(train_u, train_v)]

        return dgl.dataloading.DataLoader(
            self.g_bid,
            train_eids,
            edge_sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

    def val_dataloader(self):
        edge_sampler = dgl.dataloading.as_edge_prediction_sampler(
            self.sampler,
        )

        val_subgraph = self.g_bid.subgraph(self.val_nid)
        val_u, val_v = val_subgraph.edges()
        val_eids = val_subgraph.edata['_ID'][val_subgraph.edge_ids(val_u, val_v)]

        return dgl.dataloading.DataLoader(
            self.g_bid,
            val_eids,
            edge_sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )


@hydra.main(config_name="config", config_path="conf", version_base=None)
def train(cfg):
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cpu"
    
    train_graph_root = cfg.data.zillow_graph_root

    modal_node_ids_file = os.path.join(train_graph_root,'modal_node_ids.json')
    datamodule = DataModule(
        train_graph_root, 
        modal_node_ids_file, 
        keyword_as_src=False, 
        device=device, 
        batch_size=cfg.training.batch_size, 
        force_reload=False
    )

    model = SAGELightning(
        datamodule.in_dim,
        cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.learning_rate
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="mean_val_positive_score", save_top_k=1, mode="max"
    )

    trainer = Trainer(accelerator=device, max_epochs=cfg.training.n_epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=datamodule)

    torch.save(model.module, 'model_saved.pt')
    print('Saved Model')

@hydra.main(config_name="config", config_path="conf", version_base=None)
def evaluate(cfg):
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
    
    train_graph_root = cfg.data.zillow_graph_root

    modal_node_ids_file = os.path.join(train_graph_root,'modal_node_ids.json')
    datamodule = DataModule(
        train_graph_root, 
        modal_node_ids_file, 
        keyword_as_src=False, 
        device=device, 
        batch_size=cfg.training.batch_size, 
        force_reload=False
    )

    model = SAGELightning(
        datamodule.in_dim,
        cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.learning_rate
    )

    trainer = Trainer(accelerator=device)
    dataloader = datamodule.val_dataloader()
    trainer.test(model, dataloaders=dataloader)

@hydra.main(config_name="config", config_path="conf", version_base=None)
def baseline(cfg):
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    #root = pyrootutils.setup_root(__file__, pythonpath=True)
    
    train_graph_root = cfg.data.zillow_graph_root

    modal_node_ids_file = os.path.join(train_graph_root,'modal_node_ids.json')
    datamodule = DataModule(
        train_graph_root, 
        modal_node_ids_file, 
        keyword_as_src=False, 
        device=device, 
        batch_size=cfg.training.batch_size, 
        force_reload=False
    )
    predictor = ScorePredictor()

    mean_pos_score = MeanMetric().to(device)
    mean_neg_score = MeanMetric().to(device)
    AUROC = BinaryAUROC(thresholds=None)
    BAP = BinaryAveragePrecision(thresholds=None)

    AUROCs = []
    BAPs   = []

    dataloader = datamodule.val_dataloader()
    for input_nodes, pos_graph, neg_graph, blocks in dataloader:
        x = blocks[1].dstdata["feat"]
        # neg_graph = dgl.graph((neg_graph.edges()), num_nodes=pos_graph.num_nodes())
        pos_score = predictor(pos_graph, x)
        neg_score = predictor(neg_graph, x)

        mean_pos_score(pos_score)
        mean_neg_score(neg_score)

        scores = torch.cat([pos_score, neg_score])
        pos_label = torch.ones_like(pos_score)
        neg_label = torch.zeros_like(neg_score)
        labels = torch.cat([pos_label, neg_label])

        AUROCs.append(AUROC(scores, labels).item())
        BAPs.append(BAP(scores, labels).item())
        # break
    print("Baseline: ")
    print(f"Mean AUROC: {np.mean(AUROCs)}")
    print(f"Mean BinaryAveragePrecision: {np.mean(BAPs)}")
    print(f"Mean Positive Edge Score: {mean_pos_score.compute()}")
    print(f"Mean Negative Edge Score: {mean_neg_score.compute()}")
    print()



if __name__ == "__main__":
    train()
    # evaluate()
    # baseline()
    print("Done")
