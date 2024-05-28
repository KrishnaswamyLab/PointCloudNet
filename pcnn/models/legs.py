import numpy as np

import torch
from torch.nn import Linear
import torch.nn as nn
from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

## Authors: Alex Tong + Edward De Brouwer
## Reference: Data-Driven Learning of Geometric Scattering Networks, IEEE Machine Learning for Signal Processing Workshop 2021


class ScatterAttention(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.f_gate = nn.Sequential(nn.Linear(in_channels,in_channels),nn.ReLU(),nn.Linear(in_channels,1))
        self.f_feat = nn.Sequential(nn.Linear(in_channels,in_channels),nn.ReLU(),nn.Linear(in_channels,in_channels))
    
    def forward(self,batch, return_probs=False):
        outs = []
        probs = []
        graph_list = batch.to_data_list()
        for graph in graph_list:
            gate = self.f_gate(graph.x)
            feat = self.f_feat(graph.x)
            prob = torch.softmax(gate,0)
            outs.append(torch.sum(prob*feat,0))
            probs.append(prob)
        if return_probs:
            return torch.stack(outs), torch.stack(probs)
        else:
            return torch.stack(outs)


def scatter_moments(graph, batch, moments_returned=[1,2,3,4]):
    
    """ Compute specified statistical coefficients for each feature of each graph passed. 
        The graphs expected are disjoint subgraphs within a single graph, whose feature tensor is passed as argument "graph."
        "batch_indices" connects each feature tensor to its home graph.
        "Moments_returned" specifies the number of statistical measurements to compute. 
        If 1, only the mean is returned. If 2, the mean and variance. If 3, the mean, variance, and skew. If 4, the mean, variance, skew, and kurtosis.
        The output is a dictionary. You can obtain the mean by calling output["mean"] or output["skew"], etc.
    """
    batch_indices = batch
    # Step 1: Aggregate the features of each mini-batch graph into its own tensor
    graph_features = [torch.zeros(0).to(graph) for i in range(torch.max(batch_indices) + 1)]

    for i, node_features in enumerate(graph):
        # Sort the graph features by graph, according to batch_indices. For each graph, create a tensor whose first row is the first element of each feature, etc.
        # print("node features are", node_features)
        
        if (len(graph_features[batch_indices[i]]) == 0):  
            # If this is the first feature added to this graph, fill it in with the features.
            graph_features[batch_indices[i]] = node_features.view(-1, 1, 1)  # .view(-1,1,1) changes [1,2,3] to [[1],[2],[3]], so that we can add each column to the respective row.
        else:
            graph_features[batch_indices[i]] = torch.cat((graph_features[batch_indices[i]], node_features.view(-1, 1, 1)), dim=1)  # concatenates along columns

    statistical_moments = {"mean": torch.zeros(0).to(graph)}

    if 2 in moments_returned:
        statistical_moments["variance"] = torch.zeros(0).to(graph)
    if 3 in moments_returned:
        statistical_moments["skew"] = torch.zeros(0).to(graph)
    if 4 in moments_returned:
        statistical_moments["kurtosis"] = torch.zeros(0).to(graph)

    for data in graph_features:

        data = data.squeeze()
        
        def m(i):  # ith moment, computed with derivation data
            return torch.mean(deviation_data ** i, axis=1)

        mean = torch.mean(data, dim=1, keepdim=True)
        
        if 1 in moments_returned:
            statistical_moments["mean"] = torch.cat(
                (statistical_moments["mean"], mean.T), dim=0
            )

        # produce matrix whose every row is data row - mean of data row

        #for a in mean:
        #    mean_row = torch.ones(data.shape[1]).to( * a
        #    tuple_collect.append(
        #        mean_row[None, ...]
        #    )  # added dimension to concatenate with differentiation of rows
        # each row contains the deviation of the elements from the mean of the row
        
        deviation_data = data - mean
        
        # variance: difference of u and u mean, squared element wise, summed and divided by n-1
        variance = m(2)
        
        if 2 in moments_returned:
            statistical_moments["variance"] = torch.cat(
                (statistical_moments["variance"], variance[None, ...]), dim=0
            )

        # skew: 3rd moment divided by cubed standard deviation (sd = sqrt variance), with correction for division by zero (inf -> 0)
        skew = m(3) / (variance ** (3 / 2)) 
        skew[
            skew > 1000000000000000
        ] = 0  # multivalued tensor division by zero produces inf
        skew[
            skew != skew
        ] = 0  # single valued division by 0 produces nan. In both cases we replace with 0.
        if 3 in moments_returned:
            statistical_moments["skew"] = torch.cat(
                (statistical_moments["skew"], skew[None, ...]), dim=0
            )

        # kurtosis: fourth moment, divided by variance squared. Using Fischer's definition to subtract 3 (default in scipy)
        kurtosis = m(4) / (variance ** 2) - 3 
        kurtosis[kurtosis > 1000000000000000] = -3
        kurtosis[kurtosis != kurtosis] = -3
        if 4 in moments_returned:
            statistical_moments["kurtosis"] = torch.cat(
                (statistical_moments["kurtosis"], kurtosis[None, ...]), dim=0
            )
    
    # Concatenate into one tensor (alex)
    statistical_moments = torch.cat([v for k,v in statistical_moments.items()], axis=1)
    #statistical_moments = torch.cat([statistical_moments['mean'],statistical_moments['variance']],axis=1)
    
    return statistical_moments


class LazyLayer(torch.nn.Module):
    
    """ Currently a single elementwise multiplication with one laziness parameter per
    channel. this is run through a softmax so that this is a real laziness parameter
    """

    def __init__(self, n):
        super().__init__()
        w_dims = (2,) + n
        self.weights = torch.nn.Parameter(torch.Tensor(*w_dims))

    def forward(self, x, propogated):
        inp = torch.stack((x, propogated), dim=1)
        s_weights = torch.nn.functional.softmax(self.weights, dim=0)

        #if (len(inp.shape)-len(s_weights.shape)==2 ) and (inp.shape[-1]==1):
        #    return torch.sum(inp * s_weights[...,None], dim=1)
        #else:
        res = torch.sum(inp * s_weights, dim=1) 
        return res

    def reset_parameters(self):
        torch.nn.init.ones_(self.weights)
    

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, dtype=None):

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    
    return edge_index, deg_inv_sqrt[row] * edge_weight


class Diffuse(MessagePassing):

    """ Implements low pass walk with optional weights
    """

    def __init__(self, in_channels, out_channels, trainable_laziness=False, fixed_weights=True):

        super().__init__(aggr="add", node_dim=-3)  # "Add" aggregation.
        assert in_channels == out_channels
        self.trainable_laziness = trainable_laziness
        self.fixed_weights = fixed_weights
        if trainable_laziness:
            self.lazy_layer = LazyLayer(in_channels)
        if not self.fixed_weights:
            self.lin = torch.nn.Linear(in_channels, out_channels)


    def forward(self, x, edge_index, edge_weight=None):

        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 2: Linearly transform node feature matrix.
        # turn off this step for simplicity
        if not self.fixed_weights:
            x = self.lin(x)

        # Step 3: Compute normalization
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), dtype=x.dtype)

        # Step 4-6: Start propagating messages.
        propogated = self.propagate(edge_index, edge_weight=edge_weight, size=None, x=x)

        if not self.trainable_laziness:
            return 0.5 * (x + propogated)

        return self.lazy_layer(x, propogated)


    def message(self, x_j, edge_weight):
        
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return edge_weight.view(-1, 1, 1) * x_j


    def message_and_aggregate(self, adj_t, x):

        return matmul(adj_t, x, reduce=self.aggr)


    def update(self, aggr_out):

        # aggr_out has shape [N, out_channels]
        # Step 6: Return new node embeddings.
        return aggr_out


def feng_filters():

    tmp = np.arange(16).reshape(4,4) #tmp doesn't seem to be used!
    results = [4]
    for i in range(2, 4):
        for j in range(0, i):
            results.append(4*i+j)

    return results


class LegsFilter(torch.nn.Module):

    def __init__(self, in_channels, trainable_laziness, trainable_scales, **kwargs ):

        super().__init__()
        self.in_channels = in_channels
        self.trainable_laziness = trainable_laziness
        self.diffusion_layer1 = Diffuse((in_channels,1), (in_channels,1), trainable_laziness)
        self.diffusion_layer2 = Diffuse(
            (in_channels,4), (in_channels,4), trainable_laziness
        )
        
        if trainable_scales:
            self.wavelet_constructor = torch.nn.Parameter(torch.tensor([
                [0, -1.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1]
            ], requires_grad=True))
        else:
            self.wavelet_constructor = torch.tensor([
                [0, -1.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1]
            ], requires_grad=False) 

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        s0 = x[:,:,None]
        avgs = [s0]
        for i in range(16):
            avgs.append(self.diffusion_layer1(avgs[-1], edge_index, edge_weight = data.edge_attr))
        #for j in range(len(avgs)):
        #    avgs[j] = avgs[j][None, :, :, :]  # add an extra dimension to each tensor to avoid data loss while concatenating TODO: is there a faster way to do this?
        
        # Combine the diffusion levels into a single tensor.
        diffusion_levels = torch.stack(avgs)
        # Reshape the 3d tensor into a 2d tensor and multiply with the wavelet_constructor matrix
        # This simulates the below subtraction:
        # filter1 = avgs[1] - avgs[2]
        # filter2 = avgs[2] - avgs[4]
        # filter3 = avgs[4] - avgs[8]
        # filter4 = avgs[8] - avgs[16]
        subtracted = torch.matmul(self.wavelet_constructor, diffusion_levels.view(17, -1))
        subtracted = subtracted.view(4, x.shape[0], x.shape[1]) # reshape into given input shape
        s1 = torch.abs(
            torch.transpose(torch.transpose(subtracted, 0, 1), 1, 2))  # transpose the dimensions to match previous

        # perform a second wave of diffusing, on the recently diffused.
        avgs = [s1]
        for i in range(16): # diffuse over diffusions
            avgs.append(self.diffusion_layer2(avgs[-1], edge_index, edge_weight = data.edge_attr))
        diffusion_levels2 = torch.stack(avgs)
        
        # Having now generated the diffusion levels, we can combine them as before
        subtracted2 = torch.matmul(self.wavelet_constructor, diffusion_levels2.view(17, -1))
        subtracted2 = subtracted2.view(4, s1.shape[0], s1.shape[1], s1.shape[2])  # reshape into given input shape
        subtracted2 = torch.transpose(subtracted2, 0, 1)
        subtracted2 = torch.abs(subtracted2.reshape(-1, self.in_channels, 4))
        s2_swapped = torch.reshape(torch.transpose(subtracted2, 1, 2), (-1, 16, self.in_channels))
        s2 = s2_swapped[:, feng_filters()]

        x = torch.cat([s0, s1], dim=2)
        x = torch.transpose(x, 1, 2)
        x = torch.cat([x, s2], dim=1)
        #data.x = x.reshape(x.shape[0],-1)
        return x.reshape(x.shape[0],-1)


    def out_shape(self):

        # x * in
        return 11 * self.in_channels

    @property
    def output_dim(self):
        return self.out_shape()


class TSNet(torch.nn.Module):

    def __init__(self, in_channels, out_channels, edge_in_channels = None, trainable_laziness=False, **kwargs):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_in_channels = edge_in_channels
        self.trainable_laziness = trainable_laziness
        self.scatter = Scatter(in_channels, trainable_laziness=trainable_laziness)
        self.lin1 = Linear(self.scatter.out_shape(), out_channels)
        self.lin2 = Linear(out_channels, out_channels)
        self.lin3 = Linear(out_channels, out_channels)
        self.act = torch.nn.LeakyReLU()


    def forward(self, data):

        x, sc = self.scatter(data)
        x = self.act(x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return x, sc