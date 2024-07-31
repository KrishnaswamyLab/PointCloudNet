import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import numpy as np
import os
import pickle

def compute_dist(X):
    # computes all (squared) pairwise Euclidean distances between each data point in X
    # D_ij = <x_i - x_j, x_i - x_j>
    G = torch.matmul(X, X.T)
    D = torch.reshape(torch.diag(G), (1, -1)) + torch.reshape(torch.diag(G), (-1, 1)) - 2 * G
    return D

class GraphEnsembleLayer(nn.Module):
    def __init__(self, kernel_type, num_kernels):
        super(GraphEnsembleLayer, self).__init__()
        self.num_kernels = num_kernels
        self.epsilons = nn.Parameter(torch.ones(num_kernels, requires_grad=True))
        self.alphas = nn.Parameter(torch.ones(num_kernels,requires_grad=False))
        self.kernel_type = kernel_type

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (N, D) tensor of point cloud data
        Returns:
            graphs: list of graph data objects
        Description:
            This function computes the pairwise distance matrix between the points in the 29-dimensional point cloud
            and then constructs a graph data object for each kernel in the ensemble. The graph data
            object contains the point cloud data, the edge index, and the edge attributes
        """
        graphs = []  # List to store the graph data objects
        dist_matrix = compute_dist(point_cloud)  # Compute the pairwise distance matrix
        for i in range(self.num_kernels):
            epsilon = self.epsilons[i]
            alpha = self.alphas[i]
            if self.kernel_type == 'gaussian':
                W = torch.exp(-(dist_matrix / epsilon.pow(2)))
            if self.kernel_type == 'alpha_decay':
                W = torch.exp(-(dist_matrix / epsilon.pow(2))).pow(alpha.pow(2))
                
            edge_index, edge_attr = dense_to_sparse(torch.tensor(W))
                    
            # Create graph data object
            graph = Data(x=torch.tensor(point_cloud, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
            graphs.append(graph)
        
        return graphs

class PointCloudGraphEnsemble(nn.Module):
    def __init__(self,raw_dir, num_kernels, kernel_type):
        super(PointCloudGraphEnsemble, self).__init__()
        self.raw_dir = raw_dir
        self.graph_ensemble = GraphEnsembleLayer(kernel_type, num_kernels)
        #Load the pointclouds, patient IDs, and labels
        #Note: See test.ipynb for clarification on the structure of the point cloud dataset
        with open(os.path.join(self.raw_dir, 'pc.pkl'), 'rb') as handle:
            self.subsampled_pcs = pickle.load(handle)

        with open(os.path.join(self.raw_dir, 'patient_list.pkl'), 'rb') as handle:
            self.subsampled_patient_ids = pickle.load(handle)

        self.labels = np.load(os.path.join(self.raw_dir, 'labels.npy'))

        self.num_points = self.subsampled_pcs[0].shape[0]

    def graph_construct(self):
        data_list = []
        for i in range(len(self.subsampled_pcs)):
            point_cloud = torch.tensor(self.subsampled_pcs[i], dtype=torch.float)
            graph_ensemble = self.graph_ensemble(point_cloud)
            for graph in graph_ensemble:
                graph.patient_id = self.subsampled_patient_ids[i]
                graph.y = torch.tensor(self.labels[i], dtype=torch.long)
                data_list.append(graph)
        return data_list

