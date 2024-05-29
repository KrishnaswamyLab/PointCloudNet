import torch
from torch_geometric.data import Batch
import scipy.sparse
import numpy as np
from scipy import sparse
from pcnn.data.scattering_utils import compute_scattering_features, compute_scattering_coeffs
import torch_geometric.transforms as T
from torch_geometric.transforms import KNNGraph
import torch_geometric

from torch_geometric.transforms.base_transform import BaseTransform

def laplacian_collate_fn(batch, follow_batch = None, exclude_keys = None):

    b = Batch.from_data_list(batch, follow_batch,
                                        exclude_keys)
    
    if hasattr(batch[0],"eigvec"):
        laplacians_eigvec = [data.eigvec for data in batch]
        L_coo = scipy.sparse.block_diag(laplacians_eigvec)

        values = L_coo.data
        indices = np.vstack((L_coo.row, L_coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = L_coo.shape
        
        #L_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)) 
        b.L_i = i
        b.L_v = v
        b.L_shape = shape
        del b.eigvec 
    
    return b


def compute_dist(X):
    # computes all (squared) pairwise Euclidean distances between each data point in X
    # D_ij = <x_i - x_j, x_i - x_j>
    G = np.matmul(X, X.T)
    D = np.reshape(np.diag(G), (1, -1)) + np.reshape(np.diag(G), (-1, 1)) - 2 * G
    return D

def compute_kernel(D, eps, d):
    # computes kernel for approximating GL
    # D is matrix of pairwise distances
    K = np.exp(-D/eps) * np.power(eps, -d/2)
    return K


class laplacian_dense_transform(BaseTransform):
    def __init__(self,eps,K,d=2,eps_quantile = 0.5, fixed_pos = False, **kwargs):
        self.eps = eps
        self.K = K
        self.d = d
        self.eps_quantile = eps_quantile

        self.fixed_pos = fixed_pos

        self.node_attr_eig = None
        self.eigvec = None
        self.eps_ = None

    def forward(self,data):
        if self.fixed_pos:
            if self.edge_index is None:
                data = self.compute_eig(data)
                self.node_attr_eig = data.node_attr_eig
                self.eigvec = data.eigvec
                self.eps_ = data.eps
            else:
                data.node_attr_eig = self.node_attr_eig
                data.eigvec = self.eigvec
                data.eps = self.eps_
        else:
            data = self.compute_eig(data)
        return data 

    def compute_eig(self,data):
        # X is n x d matrix of data points
        X = data.pos.numpy()
        n = X.shape[0]
        dists = compute_dist(X)
        if self.eps == "auto":
            triu_dists = np.triu(dists)
            eps = np.quantile(triu_dists[np.nonzero(triu_dists)], self.eps_quantile)
        else:
            eps = self.eps
        W = compute_kernel(dists, eps, self.d)
        
        breakpoint()
        eps_ = data.eps
        W_ = torch_geometric.utils.to_dense_adj(data.edge_index, edge_attr = data.edge_attr)

        D = np.diag(np.sum(W, axis=1, keepdims=False))
        L = sparse.csr_matrix(D - W)
        S, U = sparse.linalg.eigsh(L, k = self.K, which='SM')
        S = np.reshape(S.real, (1, -1))/(eps * n)
        S[0,0] = 0 # manually enforce this
        # normalize eigenvectors in usual l2 norm
        U = np.divide(U.real, np.linalg.norm(U.real, axis=0, keepdims=True))

        data.node_attr_eig = torch.from_numpy(S[0])
        data.eigvec = torch.from_numpy(U)
        data.eps = eps

        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(eps={self.eps}, K={self.K},d={self.d}, eps_quantile={self.eps_quantile})'

    def __call__(self, data ):
        return self.forward(data)

def build_edge_idx(num_nodes):
    # Initialize edge index matrix
    E = torch.zeros((2, num_nodes * (num_nodes)), dtype=torch.long)
    
    # Populate 1st row
    for node in range(num_nodes):
        for neighbor in range(num_nodes):
            E[0, node * (num_nodes) + neighbor] = node

    # Populate 2nd row
    neighbors = []
    for node in range(num_nodes):
        neighbors.append(list(np.arange(node)) + list(np.arange(node, num_nodes)))
    E[1, :] = torch.Tensor([item for sublist in neighbors for item in sublist])
    
    return E

"""
def create_dense_graph(data, eps, d = 2, eps_quantile = 0.5, **kwargs):
    X = data.pos.numpy()
    n = X.shape[0]
    dists = compute_dist(X)
    if eps == "auto":
        triu_dists = np.triu(dists)
        eps = np.quantile(triu_dists[np.nonzero(triu_dists)], eps_quantile)
    W = compute_kernel(dists, eps, d)

    edge_index = build_edge_idx(n)
    edge_attr = W[edge_index[0],edge_index[1]]
    data.edge_index = edge_index
    data.edge_attr = torch.Tensor(edge_attr)
    return data
"""
    

class epsilon_graph_transform(BaseTransform):
    def __init__(self,eps, eps_quantile = 0.5,fixed_pos = False, **kwargs):
        """
        Fixed position = True will assume the same graph for all point clouds.

        Eps: is the bandwidth of the kernel. If "auto", it will be set to the quantile of the pairwise distances
        """
        self.eps = eps
        self.eps_quantile = eps_quantile
        self.fixed_pos = fixed_pos

        self.edge_index = None
        self.edge_attr = None
    
    def eta_kernel(self,x):
            return np.exp(-x)
    
    def create_epsilon_graph(self,data):
        X = data.pos.numpy()
        n = X.shape[0]
        dists = compute_dist(X)
        if self.eps == "auto":
            triu_dists = np.triu(dists)
            eps = np.quantile(triu_dists[np.nonzero(triu_dists)], self.eps_quantile)
        else:
            eps = self.eps

        W = self.eta_kernel(dists / eps)
        W[dists > eps] = 0
        
        edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(torch.Tensor(W))
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data
    
    def forward(self,data):
        if self.fixed_pos:
            if self.edge_index is None:
                data = self.create_epsilon_graph(data)
                self.edge_index = data.edge_index
                self.edge_attr = data.edge_attr
            else:
                data.edge_index = self.edge_index
                data.edge_attr = self.edge_attr
        else:
            data = self.create_epsilon_graph(data)
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(eps={self.eps}, eps_quantile={self.eps_quantile})'
    
    def __call__(self, data ):
        return self.forward(data)
    

class dense_graph_transform(BaseTransform):
    def __init__(self,eps,d=2,eps_quantile=0.5, fixed_pos = False,**kwargs):
        self.eps = eps
        self.d = d
        self.eps_quantile = eps_quantile
        self.fixed_pos = fixed_pos
        self.edge_index = None
        self.edge_attr = None

    def forward(self,data):
        if self.fixed_pos:
            if self.edge_index is None:
                data = self.compute_eig(data)
                self.edge_index = data.edge_index
                self.edge_attr = data.edge_attr
            else:
                data.edge_index = self.edge_index
                data.edge_attr = self.edge_attr
        else:
            data = self.compute_eig(data)
        return data
    
    def compute_eig(self,data):
        X = data.pos.numpy()
        n = X.shape[0]
        dists = compute_dist(X)
        if self.eps == "auto":
            triu_dists = np.triu(dists)
            eps = np.quantile(triu_dists[np.nonzero(triu_dists)], self.eps_quantile)
        else:
            eps = self.eps
        W = compute_kernel(dists, eps, self.d)

        edge_index = build_edge_idx(n)
        edge_attr = W[edge_index[0],edge_index[1]]
        data.edge_index = edge_index
        data.edge_attr = torch.Tensor(edge_attr)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(eps={self.eps}, d={self.d}, eps_quantile={self.eps_quantile})'
    
    def __call__(self, data ):
        return self.forward(data)

class scattering_features_transform(BaseTransform):
    def __init__(self,norm_list,J,scattering_n_pca, normalize_scattering_features, **kwargs):
        self.norm_list = norm_list
        self.J = J
        self.scattering_n_pca = scattering_n_pca
        self.normalize_scattering_features = normalize_scattering_features
        if self.scattering_n_pca is None:
            self.scattering_n_pca = 0
        
    def forward(self,data):
        features = compute_scattering_coeffs(data,self.norm_list,self.J)
        data.scattering_features = torch.from_numpy(features)
        return data

    def __repr__(self) -> str:

        return f'{self.__class__.__name__}(norm_list={self.norm_list}, J={self.J}, nPCA={self.scattering_n_pca}, normalize_scattering_features={self.normalize_scattering_features})'
    
    def __call__(self, data ):
        return self.forward(data)



class lap_transform(BaseTransform):
    """
    Computing the laplacian from a graph and storing the eigenvalues and eigenvectors
    """
    def __init__(self, fixed_pos, K, **kwargs):
        self.fixed_pos = fixed_pos
        self.K = K # number of eigenvalues to compute
        self.node_attr_eig = None
        self.eigvec = None

        self.eps = 1

    def compute_eigs(self,data):
        L_edge, L_vals = torch_geometric.utils.get_laplacian(data.edge_index, edge_weight = data.edge_attr)
        L_sparse = torch_geometric.utils.to_scipy_sparse_matrix(L_edge, edge_attr=L_vals, num_nodes = data.pos.shape[0])
        n = data.pos.shape[0]
        try:
            S, U = sparse.linalg.eigsh(L_sparse, k = self.K, which='SM')
        except:
            S, U = sparse.linalg.eigsh(L_sparse, k = self.K, which='SM')
        S = np.reshape(S.real, (1, -1)) /(self.eps * n)
        S[0,0] = 0 # manually enforce this
        # normalize eigenvectors in usual l2 norm
        U = np.divide(U.real, np.linalg.norm(U.real, axis=0, keepdims=True))

        data.node_attr_eig = torch.from_numpy(S[0])
        data.eigvec = torch.from_numpy(U)
        data.eps = 1

        return data 
    
    def forward(self,data):

        if self.fixed_pos:
            if self.node_attr_eig is not None:
                data.node_attr_eig = self.node_attr_eig
                data.eigvec = self.eigvec
                data.eps = self.eps
            else:
                data = self.compute_eigs(data)
                self.node_attr_eig = data.node_attr_eig
                self.eigvec = data.eigvec
        else:
            data = self.compute_eigs(data)
        return data        

    def __call__(self, data ):
        return self.forward(data)


def get_pretransforms(compute_laplacian, graph_type, compute_scattering_feats, pre_transforms_base = None, fixed_pos = False, **kwargs):

    if pre_transforms_base is None:
        pre_transforms = []
    else:
        pre_transforms = pre_transforms_base
        # T.NormalizeScale(), T.SamplePoints(display_sample)
    # scattering
    if graph_type == "knn":
        pre_transforms = pre_transforms + [ KNNGraph(kwargs["k"]) ]
    elif graph_type == "dense":
        pre_transforms = pre_transforms + [ dense_graph_transform(fixed_pos = fixed_pos, **kwargs) ]
    elif graph_type == "epsilon":
        pre_transforms = pre_transforms + [ epsilon_graph_transform(fixed_pos = fixed_pos, **kwargs) ]
    elif graph_type == "raw":
        pre_transforms = pre_transforms
    
    if compute_laplacian == "dense":
        pre_transforms = pre_transforms + [ laplacian_dense_transform(fixed_pos = fixed_pos, **kwargs)]
    elif compute_laplacian == "combinatorial":
        pre_transforms = pre_transforms + [ lap_transform(fixed_pos = fixed_pos, **kwargs) ]
     
    if compute_scattering_feats:
        pre_transforms = pre_transforms + [ scattering_features_transform(**kwargs)]

    return pre_transforms

    #GCN
    #graph_type = "knn"
    #compute_laplacian = False
    #compute_scattering_feats = False
    #pre_transforms = [ KNNGraph(kwargs["k"])  ]

    #MNN
    #graph_type = "knn"
    #compute_laplacian = "combinatorial"
    #compute_scattering_feats = False
    #pre_transforms = [ KNNGraph(kwargs["k"]), lap_transform ]

    #Scattering
    #graph_type = null
    #compute_laplacian = "epsilon"
    #compute_scattering_feats = True

