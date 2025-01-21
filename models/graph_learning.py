import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import os.path as osp
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from models.GWT import GraphWaveletTransform
from models.SWT import SimplicialWaveletTransform
import gc
gc.enable()


def compute_dist(X):
    # computes all (squared) pairwise Euclidean distances between each data point in X
    # D_ij = <x_i - x_j, x_i - x_j>
    G = torch.matmul(X, X.T)
    D = torch.reshape(torch.diag(G), (1, -1)) + torch.reshape(torch.diag(G), (-1, 1)) - 2 * G
    return D
    
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
            X_bar = (point_cloud)*self.alphas[i]
            W = compute_dist(X_bar)
            W = torch.exp(-(W / 10))
            W = torch.where(W < self.threshold, torch.zeros_like(W), W)
            gwt = GraphWaveletTransform(W, X_bar, self.device)
            PSI.append(gwt.generate_timepoint_feature())
        return torch.cat(PSI, dim = 1)

class SimplicialFeatLearningLayer(nn.Module):
    def __init__(self, n_weights, dimension, threshold, device):
        super(SimplicialFeatLearningLayer, self).__init__()
        self.alphas = nn.Parameter(torch.ones((n_weights, dimension), requires_grad=True).to(device))
        self.n_weights = n_weights
        self.threshold = threshold
        self.device = device

    def forward(self, point_cloud, eps, output_size = None):
        PSI = []
        # W = torch.square(torch.cdist(point_cloud*self.alphas, point_cloud*self.alphas))
        for i in range(self.n_weights):
            X_bar = (point_cloud)*self.alphas[i]
            W = compute_dist(X_bar)
            W = torch.exp(-(W / 0.01))
            # W = 1/(W + eps)
            # W[W==(1/eps)] = 0
            import pdb; pdb.set_trace()
            W[W<self.threshold] = 0
            swt = SimplicialWaveletTransform(W, X_bar, self.threshold, self.device)
            PSI.append(swt.calculate_wavelet_coeff(3, output_size))
        return torch.cat(PSI)
        
class PointCloudFeatLearning(nn.Module):
    def __init__(self, raw_dir, full, task, n_weights, threshold, device):
        super(PointCloudFeatLearning, self).__init__()
        self.raw_dir = raw_dir
        if self.raw_dir == "melanoma_data_full":
            if full:
                suffix = '_full'
            else:
                suffix = ''
            with open(os.path.join(self.raw_dir, 'pc'+suffix+'.pkl'), 'rb') as handle:
                self.subsampled_pcs = pickle.load(handle)
                # self.subsampled_pcs = [torch.tensor(self.subsampled_pcs[i], dtype=torch.float).to(device) for i in range(len(self.subsampled_pcs))]
                self.subsampled_pcs = [torch.tensor(StandardScaler().fit_transform(self.subsampled_pcs[i]), dtype=torch.float) for i in range(len(self.subsampled_pcs))]
            self.labels = np.load(os.path.join(self.raw_dir, 'labels'+suffix+'.npy'))
            self.num_labels = len(np.unique(self.labels))
        elif self.raw_dir == "COVID_data":
            with open(os.path.join(self.raw_dir, 'pc_covid.pkl'), 'rb') as handle:
                self.subsampled_pcs = pickle.load(handle)
                # self.subsampled_pcs = [torch.tensor(self.subsampled_pcs[i], dtype=torch.float).to(device) for i in range(len(self.subsampled_pcs))]
                self.subsampled_pcs = [torch.tensor(StandardScaler().fit_transform(self.subsampled_pcs[i]), dtype=torch.float) for i in range(len(self.subsampled_pcs))]
            with open(os.path.join(self.raw_dir, 'patient_list_covid.pkl'), 'rb') as handle:
                self.subsampled_patient_ids = pickle.load(handle)
            self.labels = np.load(os.path.join(self.raw_dir, 'labels.npy'))
            self.num_labels = len(np.unique(self.labels))
        elif raw_dir == "pdo_data":
            with open(osp.join(raw_dir, 'pc_pdo_treatment.pkl'), 'rb') as handle:
                self.subsampled_pcs = pickle.load(handle)
                # subsampled_pcs = [torch.tensor(subsampled_pcs[i], dtype=torch.float).to(device) for i in range(len(subsampled_pcs))]
                self.subsampled_pcs = [torch.tensor(StandardScaler().fit_transform(self.subsampled_pcs[i]), dtype=torch.float) for i in range(len(self.subsampled_pcs))]
            le = LabelEncoder()
            self.labels = le.fit_transform(np.load(osp.join(raw_dir, 'labels_pdo_treatment.npy')))
            self.num_labels = len(np.unique(self.labels))
            
        self.dimension = self.subsampled_pcs[0].shape[1]
        self.graph_feat = GraphFeatLearningLayer(n_weights, self.dimension, threshold, device)
        self.input_dim = self.graph_feat(self.subsampled_pcs[0].to(device), 0.000001).shape[1]
        self.device = device
    
    def forward(self, batch, eps):
        PSI = []
        for i in batch:
            psi = self.graph_feat(self.subsampled_pcs[i].to(self.device), eps)
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
        X = self.bn(X)
        for i in range(len(self.layers)-1):
            X = F.relu(self.layers[i](X))
        return self.layers[-1](X)