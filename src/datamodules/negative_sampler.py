from curses import KEY_SAVE
from pickle import FALSE
import torch as th
import dgl

import json


class NegativeSampler(object):
    def __init__(self, g, k, max_img_id, neg_share=False):
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k
        self.neg_share = neg_share
        self.max_img_id = max_img_id

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
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
