import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models.GWT import GraphWaveletTransform
import gc
gc.enable()


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
    
class GraphFeatLearningLayer(nn.Module):
    def __init__(self, n_weights, dimension, threshold, device):
        super(GraphFeatLearningLayer, self).__init__()
        self.alphas = nn.Parameter(torch.rand((n_weights, dimension), requires_grad=True).to(device))
        self.n_weights = n_weights
        self.threshold = threshold
        self.device = device

    def forward(self, point_cloud, eps):
        PSI = []
        # W = torch.square(torch.cdist(point_cloud*self.alphas, point_cloud*self.alphas))
        for i in range(self.n_weights):
            W = compute_dist((point_cloud)*self.alphas[i])
            W = 1/(W + eps)
            W[W==(1/eps)] = 0
            W[W<self.threshold] = 0
            gwt = GraphWaveletTransform(W, point_cloud, self.device)
            PSI.append(gwt.generate_timepoint_feature())
        return torch.cat(PSI, dim = 1)
        
class PointCloudFeatLearning(nn.Module):
    def __init__(self, raw_dir, full, n_weights, threshold, device):
        super(PointCloudFeatLearning, self).__init__()
        self.raw_dir = raw_dir
        if full:
            suffix = '_full'
        else:
            suffix = ''
        with open(os.path.join(self.raw_dir, 'pc'+suffix+'.pkl'), 'rb') as handle:
            self.subsampled_pcs = pickle.load(handle)
            # self.subsampled_pcs = [torch.tensor(self.subsampled_pcs[i], dtype=torch.float).to(device) for i in range(len(self.subsampled_pcs))]
            self.subsampled_pcs = [torch.tensor(StandardScaler().fit_transform(self.subsampled_pcs[i]), dtype=torch.float).to(device) for i in range(len(self.subsampled_pcs))]
        with open(os.path.join(self.raw_dir, 'patient_list'+suffix+'.pkl'), 'rb') as handle:
            self.subsampled_patient_ids = pickle.load(handle)
        self.labels = np.load(os.path.join(self.raw_dir, 'labels'+suffix+'.npy'))
        self.dimension = self.subsampled_pcs[0].shape[1]
        self.graph_feat = GraphFeatLearningLayer(n_weights, self.dimension, threshold, device)
        self.num_labels = len(np.unique(self.labels))
        self.input_dim = self.graph_feat(self.subsampled_pcs[0], 0.01).shape[1]
        self.device = device
    
    def forward(self, batch, eps):
        PSI = []
        for i in batch:
            psi = self.graph_feat(self.subsampled_pcs[i], eps)
            PSI.append(psi.mean(0))
            del(psi)
            torch.cuda.empty_cache()
            gc.collect()
        return torch.stack(PSI, dim=0)
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
        # X = self.bn(X)
        for i in range(len(self.layers)-1):
            X = F.relu(self.layers[i](X))
        return self.layers[-1](X)