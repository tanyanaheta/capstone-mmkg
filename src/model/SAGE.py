from curses import pair_content
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl import AddSelfLoop
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
    as_edge_prediction_sampler,
)

from dgl.nn.pytorch.conv import SAGEConv


class SAGE(nn.Module):
    def __init__(self, in_dim, h_dim, n_layers, activation, dropout, sage_conv_method):
        super().__init__()
        self.init(in_dim, h_dim, n_layers, activation, dropout, sage_conv_method)

    def init(
        self, in_dim, h_dim, out_dim, n_layers, activation, dropout, sage_conv_method
    ):
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.out_dim = out_dim

        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_dim, h_dim, sage_conv_method))
        for i in range(1, n_layers):
            self.layers.append(dglnn.SAGEConv(h_dim, h_dim, sage_conv_method))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate((self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
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


# def train_graph(g, features, masks, model):
#     # define train/val samples, loss function and optimizer
#     train_mask, val_mask = masks
#     loss_fcn = torch.nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

#     # training loop
#     initial_distance = 0
#     for epoch in range(500):
#         model.train()
#         logits = model(g, features)
#         loss = loss_fcn(logits[train_mask], g.ndata["feat"][train_mask])
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         acc = evaluate(g, features, g.ndata["feat"], val_mask, model)

#         if initial_distance == 0:
#             initial_distance = acc

#         if epoch % 25 == 0:
#             print(
#                 "Epoch {:05d} | Loss {:.4f} | Distance Reduced {:.4f} %".format(
#                     epoch, loss.item(), (initial_distance - acc) / initial_distance
#                 )
#             )


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


# def run(g, sage_conv_method="mean"):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     g = g.int().to(device)
#     features = g.ndata["feat"]
#     masks = g.ndata["train_mask"], g.ndata["val_mask"]

#     # create GraphSAGE model
#     in_size = features.shape[1]
#     out_size = features.shape[1]
#     model = SAGE(in_size, None, out_size, sage_conv_method).to(device)

#     # model training
#     print("Training...")
#     train_graph(g, features, masks, model)

#     # test the model
#     print("Testing...")
#     acc = evaluate(g, features, g.ndata["feat"], g.ndata["test_mask"], model)
#     print("Testing Complete")

#     return model.forward(g, g.ndata["feat"])


def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata["train_mask"])
    val_g = g.subgraph(g.ndata["train_mask"] | g.ndata["val_mask"])
    test_g = g
    return train_g, val_g, test_g


# def train(args, device, g, reverse_eids, seed_edges, model):
#     # create sampler & dataloader
#     sampler = NeighborSampler([15, 10, 5], prefetch_node_feats=["feat"])
#     sampler = dgl.dataas_edge_prediction_sampler(
#         sampler,
#         exclude="reverse_id",
#         reverse_eids=reverse_eids,
#         negative_sampler=negative_sampler.Uniform(1),
#     )
#     use_uva = args.mode == "mixed"
#     dataloader = DataLoader(
#         g,
#         seed_edges,
#         sampler,
#         device=device,
#         batch_size=512,
#         shuffle=True,
#         drop_last=False,
#         num_workers=0,
#         use_uva=use_uva,
#     )
#     opt = torch.optim.Adam(model.parameters(), lr=0.0005)
#     for epoch in range(10):
#         model.train()
#         total_loss = 0
#         for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
#             dataloader
#         ):
#             x = blocks[0].srcdata["feat"]
#             pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
#             score = torch.cat([pos_score, neg_score])
#             pos_label = torch.ones_like(pos_score)
#             neg_label = torch.zeros_like(neg_score)
#             labels = torch.cat([pos_label, neg_label])
#             loss = F.binary_cross_entropy_with_logits(score, labels)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#             total_loss += loss.item()
#             if (it + 1) == 1000:
#                 break
#         print("Epoch {:05d} | Loss {:.4f}".format(epoch, total_loss / (it + 1)))
