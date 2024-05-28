import torch.nn as nn
import torch_geometric
import torch
import numpy as np


def compute_dist(X):
    # computes all (squared) pairwise Euclidean distances between each data point in X
    # D_ij = <x_i - x_j, x_i - x_j>
    G = np.matmul(X, X.T)
    D = np.reshape(np.diag(G), (1, -1)) + np.reshape(np.diag(G), (-1, 1)) - 2 * G
    return D

class GraphLearningLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.epsilon = nn.Parameter(torch.ones(1, requires_grad=True))

        self.num_edges = 10 #maximum number of edges in the graph
        #self.alpha = nn.Parameter(torch.ones(1,requires_grad=False))
        #self.alpha = 1 

    def forward(self,data):
        d_list = []
        i_b = 0
        for i_b in data.batch.unique():
            cdists = torch.cdist(data.pos[data.batch==i_b][None], data.pos[data.batch==i_b][None],2)[0]
            W = torch.exp(-(cdists / self.epsilon.pow(2)))#.pow(self.alpha.pow(2)))
            #mask = W<self.epsilon
            #W = W * mask
            
            mask = (torch.arange(len(W))[None] != torch.arange(len(W))[:,None]).to(W.device)
            W = W * mask

            #Sampling edges according to their weights
            Ws = torch.softmax(W,axis=1)
            ps = Ws.cumsum(1)
            idx_target = torch.searchsorted(ps,torch.rand(len(W),self.num_edges).to(W.device))
            if idx_target.max() >= len(W):
                breakpoint()

            idx_ref = (torch.arange(len(W))[:,None].repeat(1,self.num_edges)).to(W.device)
            edge_index = torch.stack([idx_ref.flatten(),idx_target.flatten()],dim=0)
            edge_attr = W[edge_index[0],edge_index[1]]

            #edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(W)
            d_list.append(torch_geometric.data.Data(edge_index = edge_index, edge_attr = edge_attr, num_nodes = W.shape[0]))
        
        d_batch = torch_geometric.data.Batch.from_data_list(d_list)
        data.edge_index = d_batch.edge_index
        data.edge_attr = d_batch.edge_attr
        return data

