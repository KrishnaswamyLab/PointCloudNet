import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import defaultdict

from models.GWT import GraphWaveletTransform
from models.SWT import SimplicialWaveletTransform
import gc
gc.enable()


def compute_dist(X):
    G = torch.matmul(X, X.T)
    D = torch.reshape(torch.diag(G), (1, -1)) + torch.reshape(torch.diag(G), (-1, 1)) - 2 * G
    return D
class GraphFeatLearningLayer(nn.Module):
    def __init__(self, n_weights, dimension, threshold, device):
        super().__init__()
        self.alphas = nn.Parameter(torch.rand((n_weights, dimension), requires_grad=True).to(device))
        self.n_weights = n_weights
        self.threshold = threshold
        self.device = device

    def forward(self, point_clouds, sigma):
        B_pc = len(point_clouds)
        d = point_clouds[0].shape[1]
        
        all_edge_indices = []
        all_edge_weights = []
        all_node_feats   = []
        
        batch = []
        batch_pc   = []

        node_offset = 0 

        for p in range(B_pc):
            pc = point_clouds[p] 
            num_points = pc.shape[0] 
            for i in range(self.n_weights):
                X_bar = pc * self.alphas[i]
                
                W = compute_dist(X_bar)
                W = torch.exp(-W / sigma)
                W = torch.where(W < self.threshold, torch.zeros_like(W), W)
                d = W.sum(0)
                W = W/d
                # W[torch.isnan(W)] = 0
                # W = 1/2*(torch.eye(W.shape[0]).to(self.device)+W)
                
                row, col = torch.where(W > 0)
                w_vals   = W[row, col]

                row_offset = row + node_offset
                col_offset = col + node_offset
                all_edge_indices.append(torch.cat([torch.stack([row_offset, col_offset], dim=0), torch.arange(W.shape[0]).repeat(2,1).to(W.device)], 1))
                all_edge_weights.append(torch.cat([w_vals/2, 0.5*torch.ones(W.shape[0]).to(W.device)]))

                all_node_feats.append(X_bar)

                batch.extend([p*self.n_weights + i]*num_points)

                node_offset += num_points

        edge_index = torch.cat(all_edge_indices, dim=1).to(self.device)  
        edge_weight = torch.cat(all_edge_weights, dim=0).to(self.device) 
        X_cat = torch.cat(all_node_feats, dim=0).to(self.device)

        batch = torch.tensor(batch, device=self.device, dtype=torch.long)

        J = 3
        gwt = GraphWaveletTransform(edge_index, edge_weight, X_cat, J, self.device)

        features = gwt.diffusion_only(batch)
        return features.view(B_pc, features.shape[1] * self.n_weights)
class SimplicialFeatLearningLayer(nn.Module):
    def __init__(self, n_weights, dimension, threshold, device):
        super().__init__()
        # shape = [n_weights, dimension], each row i is alpha_i \in R^dimension
        self.alphas = nn.Parameter(torch.rand((n_weights, dimension), requires_grad=True).to(device))
        self.n_weights = n_weights
        self.threshold = threshold
        self.device = device

    def forward(self, point_clouds, sigma):
        B_pc = len(point_clouds)
        dim = point_clouds[0].shape[1]

        all_edge_indices = []
        all_edge_weights = []
        all_features     = []

        batch = []

        node_offset = 0
        self.indices = []

        for p in range(B_pc):
            pc = point_clouds[p]  # [N_pts, dim]
            N_pts = pc.shape[0]
            for w in range(self.n_weights):
                alpha_w = self.alphas[w]  # shape [dim]
                X_nodes = pc * alpha_w    # shape [N_pts, dim]

                W = compute_dist(X_nodes)    # [N_pts, N_pts]
                W = torch.exp(-W / sigma)
                # W = torch.where(W < self.threshold, torch.zeros_like(W), W)

                i_idx, j_idx = torch.where(W >= self.threshold)
                all_edge_indices.append(torch.stack([i_idx, j_idx]))
                edge_weights_ij = W[i_idx, j_idx]
                all_edge_weights.append(edge_weights_ij)
                # mask_ij = (i_idx < j_idx)
                # i_idx = i_idx[mask_ij]
                # j_idx = j_idx[mask_ij]
                # edge_weights_ij = edge_weights_ij[mask_ij]
                num_edges = i_idx.shape[0]

                # potential_tri = torch.combinations(torch.arange(N_pts, device=self.device), r=3)
                # i_t, j_t, k_t = potential_tri.T
                # tri_mask = (
                #     (W[i_t, j_t] > self.threshold) &
                #     (W[j_t, k_t] > self.threshold) &
                #     (W[i_t, k_t] > self.threshold)
                # )
                # valid_tri = potential_tri[tri_mask]  # shape [?, 3]
                # num_tri = valid_tri.size(0)
                W_thresh = (W >= self.threshold)  # bool matrix of shape [N_pts, N_pts]
                neighbors = [set() for _ in range(N_pts)]

                # Gather edges i->j and j->i in neighbors
                # Because W_thresh is likely symmetric for distances, we ensure (i < j) for edges
                i_idx, j_idx = torch.where(W_thresh)
                for i, j in zip(i_idx.tolist(), j_idx.tolist()):
                    # Optional: keep edges only for i<j to avoid duplicates
                    if i < j:
                        neighbors[i].add(j)
                        neighbors[j].add(i)

                # 2) For each edge (i, j), find common neighbors (intersection)
                triangles = []
                for i in range(N_pts):
                    for j in neighbors[i]:
                        # Only look "forward" j > i to avoid duplicates or reversed edges
                        if j > i:
                            common_neighbors = neighbors[i].intersection(neighbors[j])
                            # Again, pick k > j to avoid duplicates
                            for k in common_neighbors:
                                if k > j:
                                    triangles.append((i, j, k))

                valid_tri = torch.tensor(triangles, device=self.device)[:1000]  # shape [?, 3]
                num_tri = valid_tri.size(0)

                # # 3) Compute triangle centroids (if needed)
                # X_tri = (X_nodes[valid_tri[:, 0]] +
                #         X_nodes[valid_tri[:, 1]] +
                #         X_nodes[valid_tri[:, 2]]) / 3.0

                X_edges = 0.5 * ( X_nodes[i_idx] + X_nodes[j_idx] )
                if(num_tri):
                    X_tri = (X_nodes[valid_tri[:,0]] +
                            X_nodes[valid_tri[:,1]] +
                            X_nodes[valid_tri[:,2]]) / 3.0
                    X_bar = torch.cat([X_nodes, X_edges, X_tri], dim=0)  
                else:
                    X_bar = torch.cat([X_nodes, X_edges], dim=0)  
                index = {}
                edges = torch.stack((i_idx,j_idx)).T
                for k,v in enumerate(edges.tolist()):
                    index[frozenset(v)] = k

                edge_pairs = []
                for e1 in index.keys():
                    for e2 in index.keys():
                        if(len(e1.intersection(e2)) == 1):
                            edge_pairs.append([index[e1], index[e2]])
                            edge_pairs.append([index[e2], index[e1]])
                
                index = {}
                for k,v in enumerate(valid_tri.tolist()):
                    index[frozenset(v)] = k
                tri_pairs = []
                for t1 in index.keys():
                    for t2 in index.keys():
                        if(len(t1.intersection(t2)) == 2):
                            tri_pairs.append([index[t1], index[t2]])
                            tri_pairs.append([index[t2], index[t1]])

                base_nodes = node_offset
                base_edges = node_offset + N_pts
                base_tris  = node_offset + N_pts + num_edges
                edge_pairs_tensor = torch.tensor(edge_pairs, dtype=torch.long, device=self.device)
                edge_pairs_tensor = torch.unique(edge_pairs_tensor, dim=0)
                all_edge_indices.append(edge_pairs_tensor.T + base_edges)
                all_edge_weights.append(edge_weights_ij[edge_pairs_tensor.T[0]] + edge_weights_ij[edge_pairs_tensor.T[1]])

                # all_edge_weights.append(edge_weights_ij[edge_pairs.T[0]] + edge_weights_ij[edge_pairs.T[1]])
                if(num_tri):
                    tri_pairs_tensor = torch.tensor(tri_pairs, dtype=torch.long, device=self.device)
                    all_edge_indices.append(tri_pairs_tensor.T + base_tris)
                    all_edge_weights.append(torch.ones(len(tri_pairs), dtype=torch.float, device=self.device))
                all_features.append(X_bar)

                n_total = N_pts + num_edges + num_tri
                batch.extend([p*self.n_weights + w]*n_total)

                node_offset += n_total

        edge_index = []
        edge_weight = []
        for i, w in zip(all_edge_indices, all_edge_weights):
            edge_index.append(i)
            edge_weight.append(w)

        edge_index_cat = torch.cat(edge_index, dim=1) if len(edge_index)>0 else torch.empty((2,0), device=self.device)
        edge_weight_cat = torch.cat(edge_weight, dim=0) if len(edge_weight)>0 else torch.empty((0,), device=self.device)

        X_cat = torch.cat(all_features, dim=0) if all_features else torch.empty((0, dim), device=self.device)
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)

        J = 3
        gwt = GraphWaveletTransform(edge_index_cat, edge_weight_cat, X_cat, J, self.device)

        features = gwt.generate_timepoint_features(batch)
        return features.view(B_pc, features.shape[1] * self.n_weights)

class HiPoNet(nn.Module):
    def __init__(self, model, dimension, n_weights, threshold, device):
        super(HiPoNet, self).__init__()
        self.dimension = dimension
        if(model=='graph'):
            self.layer = GraphFeatLearningLayer(n_weights, dimension, threshold, device)
        else:
            self.layer = SimplicialFeatLearningLayer(n_weights, dimension, threshold, device)
        self.device = device
    
    def forward(self, batch, sigma):
        PSI = self.layer(batch, sigma)
        return PSI

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.sf = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm1d(input_dim)
        if(num_layers==1):
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            for i in range(num_layers-2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, X):
        X = self.bn(X)
        for i in range(len(self.layers)-1):
            X = F.relu(self.layers[i](X))
        return (self.layers[-1](X))