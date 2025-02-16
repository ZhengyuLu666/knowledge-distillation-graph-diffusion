__author__ = "Stefan Weißenberger and Johannes Gasteiger"
__license__ = "MIT"

import numpy as np
from typing import List
import torch.nn.functional as F
import torch
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, InMemoryDataset
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix
from layers import GraphConvolution
class GDC(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(GDC, self).__init__()

        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(GCNConv(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight=edge_attr)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return torch.nn.functional.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self,num_node_features,num_classes):
        super(GCN, self).__init__()
        # Graph convolution
        # self.conv1 = GCNConv(dataset.num_node_features, 32)
        self.conv1 = GCNConv(num_node_features, 16)
        # self.conv2 = GCNConv(32, dataset.num_classes)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Forward propagation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    # Compute nodes with high confidence in a round of self-training
    def confidence(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        softmax = F.softmax(x, dim=1)
        # log_softmax = F.log_softmax(x, dim=1)
        # comentropy = (-1) * (softmax.mul(log_softmax).sum(axis=1))
        # comentropy = comentropy.reshape(len(comentropy), 1)
        prob_max = softmax.max(dim=1)[0]
        prob_max = prob_max.reshape(len(prob_max),1)
        pre_label = softmax.argmax(dim=1)  # Classes of predictions
        pre_label = pre_label.reshape(len(pre_label), 1)

        # Confidence matrix with 3 columns  [number, comentropy, pseudo-label]
        temp = np.arange(len(data.y))
        temp = temp[:, np.newaxis]
        # new_x = np.hstack([temp, comentropy.detach().cpu().numpy()])
        new_x = np.hstack([temp, prob_max.detach().cpu().numpy()])
        new_x = np.hstack([new_x, pre_label.detach().cpu().numpy()])

        return new_x


class GCN1 (torch.nn.Module):
    def __init__(self, nfeat,  nclass, nhid:int=16, dropout:float=0.5):
        super(GCN1, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        adj = to_dense_adj(edge_index)[0]
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
        # return x

    def confidence(self, data):
        x, edge_index = data.x, data.edge_index
        adj = to_dense_adj(edge_index)[0]
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        softmax = F.softmax(x, dim=1)
        prob_max = softmax.max(dim=1)[0]
        prob_max = prob_max.reshape(len(prob_max), 1)
        pre_label = softmax.argmax(dim=1)  # Classes of predictions
        pre_label = pre_label.reshape(len(pre_label), 1)

        # Confidence matrix with 3 columns  [number, comentropy, pseudo-label]
        temp = np.arange(len(x))
        temp = temp[:, np.newaxis]
        # new_x = np.hstack([temp, comentropy.detach().cpu().numpy()])
        new_x = np.hstack([temp, prob_max.detach().cpu().numpy()])
        new_x = np.hstack([new_x, pre_label.detach().cpu().numpy()])

        return new_x

class GAT(torch.nn.Module):
    def __init__(self, nfeat, nclass, nhid:int=8, dropout:float=0.5, alpha:float=0.2, nheads:int=8):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        adj  = to_dense_adj(edge_index)[0]
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(torch.nn.Module):
    def __init__(self, nfeat, nclass, nhid:int=8, dropout:float=0.5, alpha:float=0.2, nheads:int=8):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # print("torch.where(torch.isnan(x before drop)", torch.where(torch.isnan(x) == True))
        # print("torch.where(torch.isinf(x before drop)", torch.where(torch.isinf(x) == True))
        x = F.dropout(x, self.dropout, training=self.training)
        # print("torch.where(torch.isnan(x)", torch.where(torch.isnan(x) == True))
        # print("torch.where(torch.isinf(x)", torch.where(torch.isinf(x) == True))
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)