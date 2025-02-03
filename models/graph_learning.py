import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

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

                row, col = torch.where(W >= self.threshold)
                w_vals   = W[row, col]

                row_offset = row + node_offset
                col_offset = col + node_offset

                all_edge_indices.append(torch.stack([row_offset, col_offset], dim=0))
                all_edge_weights.append(w_vals)

                all_node_feats.append(X_bar)

                batch.extend([p*self.n_weights + i]*num_points)

                node_offset += num_points

        edge_index = torch.cat(all_edge_indices, dim=1).to(self.device)  
        edge_weight = torch.cat(all_edge_weights, dim=0).to(self.device) 
        X_cat = torch.cat(all_node_feats, dim=0).to(self.device)

        batch = torch.tensor(batch, device=self.device, dtype=torch.long)

        J = 3
        gwt = GraphWaveletTransform(edge_index, edge_weight, X_cat, J, self.device)

        features = gwt.generate_timepoint_features(batch)
        return features.view(B_pc, features.shape[1] * self.n_weights)
class SimplicialFeatLearningLayer(nn.Module):
    def __init__(self, n_weights, dimension, threshold, device):
        super(SimplicialFeatLearningLayer, self).__init__()
        self.alphas = nn.Parameter(torch.ones((n_weights, dimension), requires_grad=True).to(device))
        self.n_weights = n_weights
        self.threshold = threshold
        self.device = device

    def forward(self, point_cloud, sigma, output_size = None):
        PSI = []
        for i in range(self.n_weights):
            X_bar = (point_cloud)*self.alphas[i]
            W = compute_dist(X_bar)
            W = torch.exp(-(W / sigma))
            W = torch.where(W < self.threshold, torch.zeros_like(W), W)
            swt = SimplicialWaveletTransform(W, X_bar, self.threshold, self.device)
            PSI.append(swt.calculate_wavelet_coeff(3, output_size))
        return torch.cat(PSI)

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

class HiPoNetSpaceGM(nn.Module):
    def __init__(self, raw_dir, label_name, n_weights, spatial_threshold, gene_threshold, sigma, model, device):
        super(HiPoNetSpaceGM, self).__init__()
        self.raw_dir = raw_dir

        self.sigma = sigma
        if(model=='graph'):
            self.space_encoder = GraphFeatLearningLayer(1, 2, spatial_threshold, device)
            self.gene_encoder = GraphFeatLearningLayer(1, self.gene_dim, gene_threshold, device)
        else:
            self.space_encoder = SimplicialFeatLearningLayer(1, 2, spatial_threshold, device)
            self.gene_encoder = SimplicialFeatLearningLayer(1, self.gene_dim, gene_threshold, device)
        with torch.no_grad():
            self.input_dim = self.space_encoder(self.spatial_cords[0].to(device), self.sigma).shape[0] + self.gene_encoder(self.gene_expr[0].to(device), 10).shape[0]
        self.device = device
        self.num_labels = 2
    
    def forward(self, batch, eps):
        PSI_spatial = []
        PSI_gene = []
        for i in batch:
            psi_spatial = self.space_encoder(self.spatial_cords[i].to(self.device), self.sigma)
            psi_gene = self.gene_encoder(self.gene_expr[i].to(self.device), 10)
            PSI_spatial.append(psi_spatial)#.mean(0))
            PSI_gene.append(psi_gene)#.mean(0))
            del(psi_spatial, psi_gene)
            torch.cuda.empty_cache()
            gc.collect()
        return torch.cat((torch.stack(PSI_spatial, dim=0), torch.stack(PSI_gene, dim=0)),1)

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