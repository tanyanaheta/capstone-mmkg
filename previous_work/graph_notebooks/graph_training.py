import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

from dgl.nn.pytorch.conv import SAGEConv


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, sage_conv_method):
        super().__init__()
        self.layers = nn.ModuleList()
        # one-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, out_size, sage_conv_method))
        # self.layers.append(dglnn.SAGEConv(hid_size, out_size, sage_conv_method))
        self.dropout = nn.Dropout(0.0)

    def forward(self, graph, x):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        preds = model(g, features)
        preds = preds[mask]
        labels = labels[mask]

        euc_distance = (preds - labels).pow(2).sum().sqrt()
        return euc_distance


def train_graph(g, features, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask, val_mask = masks
    loss_fcn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    initial_distance = 0
    for epoch in range(500):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], g.ndata['feat'][train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, g.ndata['feat'], val_mask, model)

        if initial_distance == 0:
            initial_distance = acc

        if epoch % 25 == 0:
            print(
                "Epoch {:05d} | Loss {:.4f} | Distance Reduced {:.4f} %".format(
                    epoch, loss.item(), (initial_distance - acc) / initial_distance
                )
            )

def run(g, sage_conv_method='mean'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"]

    # create GraphSAGE model
    in_size = features.shape[1]
    out_size = features.shape[1]
    model = SAGE(in_size, None, out_size, sage_conv_method).to(device)

    # model training
    print("Training...")
    train_graph(g, features, masks, model)

    # test the model
    print("Testing...")
    acc = evaluate(g, features, g.ndata['feat'], g.ndata["test_mask"], model)
    print("Testing Complete")

    return model.forward(g, g.ndata['feat'])