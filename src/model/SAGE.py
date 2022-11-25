import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import tqdm
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
    as_edge_prediction_sampler,
)
from dgl.nn.pytorch.conv import SAGEConv
from pytorch_lightning import LightningDataModule, LightningModule, Trainer


class SAGE(nn.Module):
    def __init__(
        self,
        in_dim,
        h_dim=512,
        n_layers=3,
        activation=F.relu,
        dropout=0.1,
        sage_conv_method="mean",
    ):
        super().__init__()
        self.init(in_dim, h_dim, n_layers, activation, dropout, sage_conv_method)

    def init(self, in_dim, h_dim, n_layers, activation, dropout, sage_conv_method):
        self.n_layers = n_layers
        self.h_dim = h_dim

        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.SAGEConv(
                in_dim, 
                h_dim, 
                sage_conv_method, 
                feat_drop=dropout, 
                norm=partial(F.normalize, p=2, dim=-1),
                activation=activation
            )
        )
        for i in range(1, n_layers):
            self.layers.append(
                    dglnn.SAGEConv(
                        in_dim, 
                        h_dim, 
                        sage_conv_method, 
                        feat_drop=dropout, 
                        norm=partial(F.normalize, p=2, dim=-1),
                        activation=activation                        
                    )
                )

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
        return h

    def inference(
        self,
        g,
        x,
        batch_size,
        device,
        num_workers,
    ):
        """
        -- WIP --

        Inference with the GraphSAGE model on sampled neighbors
        """

        feat = g.ndata["feat"]

        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device
        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.h_dim, device=buffer_device, pin_memory=pin_memory
            )

            feat = feat.to(device)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                dataloader, desc="Inference"
            ):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.to(buffer_device)

            feat = y
        return y


# def evaluate(g, features, labels, mask, model):
#     model.eval()
#     with torch.no_grad():
#         preds = model(g, features)
#         preds = preds[mask]
#         labels = labels[mask]

#         euc_distance = (preds - labels).pow(2).sum().sqrt()
#         return euc_distance


# def compute_mrr(model, evaluator, node_emb, src, dst, neg_dst, device, batch_size=500):
#     rr = torch.zeros(src.shape[0])
#     for start in tqdm.trange(0, src.shape[0], batch_size, desc="Evaluate"):
#         end = min(start + batch_size, src.shape[0])
#         all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
#         h_src = node_emb[src[start:end]][:, None, :].to(device)
#         h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
#         pred = model.predictor(h_src * h_dst).squeeze(-1)
#         input_dict = {"y_pred_pos": pred[:, 0], "y_pred_neg": pred[:, 1:]}
#         rr[start:end] = evaluator.eval(input_dict)["mrr_list"]
#     return rr.mean()


def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata["train_mask"])
    val_g = g.subgraph(g.ndata["train_mask"] | g.ndata["val_mask"])
    test_g = g
    return train_g, val_g, test_g
