import os

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
            print(edge_subgraph)
            edge_subgraph.apply_edges(fn.u_dot_v("feat", "feat", "score"))
            return edge_subgraph.edata["score"]


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
        dataset = dgl.data.CSVDataset(csv_dataset_root, force_reload=force_reload)
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
def baseline(cfg):
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    root = pyrootutils.setup_root(__file__, pythonpath=True)

    datamodule = DataModule(
        root / cfg.data.zillow.path, device=device, batch_size=cfg.training.batch_size
    )
    predictor = ScorePredictor()

    dataloader = datamodule.val_dataloader()
    for input_nodes, pos_graph, neg_graph, blocks in dataloader:
        x = blocks[0].srcdata["feat"]
        print(x)
        pos_score = predictor(pos_graph, x)
        neg_score = predictor(neg_graph, x)
        print(pos_score)
        print(neg_score)


if __name__ == "__main__":
    baseline()
    print("Done")
