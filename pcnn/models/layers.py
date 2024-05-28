import torch
import torch.nn as nn
import torch_geometric
import pytorch_lightning as pl
from pcnn.models.legs import LegsFilter

    
class BaseLayer(nn.Module):
    def __init__(self, filter_method,
                combine_method,
                activation,
                cross_channel_conv,
                reshaping,
                input_dim, # number of input channels
                num_filters, 
                num_combine,
                num_cross_channels,
                output_dim,
                num_scattering_feats = 0, **kwargs):
        super(BaseLayer, self).__init__()

        self.input_dim = input_dim
        self.num_filters = num_filters # corresponds to J_l in the paper (for GCN = 1)
        self.num_combine = num_combine # corresponds to C_l' in the paper (for GCN = input_dim)
        self.num_cross_channels = num_cross_channels # corresponds to J_l' in the paper (for GCN = 1)
        self.output_dim = output_dim
        self.num_scattering_feats = num_scattering_feats


        self.filtering_layer = self._get_filtering_layer(filter_method, **kwargs)
        self.combine_layer = self._get_combine_layer(combine_method)
        self.activation = self._get_activation(activation)
        self.cross_channel_conv = self._get_cross_channel_conv(cross_channel_conv)
        self.reshaping = self._get_reshaping(reshaping)

        if filter_method == "extract_scattering":
            self.output_dim = num_scattering_feats

    def forward(self, x):
        x_ = self.filtering_layer(x)
        x_ = self.combine_layer(x_)
        x_ = self.activation(x_)
        x_ = self.cross_channel_conv(x_)
        x_ = self.reshaping(x_)

        x.x = x_
        return x
    
    def _get_filtering_layer(self, filter_method, **kwargs):
        if filter_method == 'gcn':
            return GCNFilter(input_dim = self.input_dim, num_filters = self.num_filters)
        elif filter_method == 'mnn_diffusion':
            return MNNDiffusionFilter(input_dim = self.input_dim, num_filters = self.num_filters, **kwargs)
        elif filter_method == "mnn":
            return MNNFilter(input_dim = self.input_dim, num_filters = self.num_filters, **kwargs)
        elif filter_method == "scattering":
            return ScatteringFilter(input_dim = self.input_dim, num_filters = self.num_filters, **kwargs)
        elif filter_method == "extract_scattering": #extract the precomputed scattering features
            self.output_dim = self.num_scattering_feats
            return lambda x: x.scattering_features.reshape(x.scattering_features.shape[0],-1)
        elif filter_method == "legs":
            filter_layer =  LegsFilter(in_channels = self.input_dim, **kwargs)
            self.output_dim = filter_layer.output_dim
            return filter_layer
    
    def _get_combine_layer(self,combine_method):
        if combine_method == "identity":
            return lambda x: x
        elif combine_method == "sum":
            return lambda x: torch.sum(x, dim = 1)

    
    def _get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "identity":
            return lambda x: x
        elif activation == "abs":
            return lambda x: torch.abs(x)
    
    def _get_cross_channel_conv(self, cross_channel_conv):
        if cross_channel_conv == "identity":
            return lambda x: x
        
    def _get_reshaping(self, reshaping):
        if reshaping == "identity":
            return lambda x: x
        elif reshaping == "flatten":
            self.output_dim = self.output_dim * self.input_dim
            return lambda x: x.reshape(x.shape[0],-1)
    

class GCNFilter(nn.Module):
    def __init__(self, input_dim, num_filters,**kwargs):
        super().__init__()

        self.mod = torch_geometric.nn.GCNConv(in_channels = input_dim, out_channels = num_filters) 

    def forward(self,x):
        return self.mod(x.x, x.edge_index, x.edge_attr)
    
class MNNDiffusionFilter(nn.Module):
    def __init__(self, input_dim, num_filters, K, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.K = K # number of expansions of the diffusion operator

        self.alpha_k = nn.Parameter(torch.randn(num_filters, K))

    def forward(self,x):
        filtered_outputs = []
        for k in range(self.K):
            filtered_outputs.append(torch.mm(x.Pk[k],x.x)[...,None] * self.alpha_k[:,k][None,None,:])
        filtered_tensor = torch.stack(filtered_outputs) # Size = [K x N x input_dim x num_filters]
        filtered_tensor = torch.sum(filtered_tensor, dim = 0) # Size = [N x input_dim x num_filters]
        return filtered_tensor
    

class ExpPolyFilter(nn.Module):
    def __init__(self,max_poly_order = 5):
        super().__init__()
        self.max_poly_order = max_poly_order
        self.alpha_k = nn.Parameter(torch.randn(max_poly_order), requires_grad = True)

    def forward(self,x):
        x_exp = torch.exp(-x)
        poly_out = torch.sum(torch.stack([x_exp.pow(i)*self.alpha_k[i] for i in range(self.max_poly_order)]), dim = 0)
        return poly_out

class MNNFilter(nn.Module):
    def __init__(self, input_dim, num_filters, poly_filter = False, scattering_filter = False, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.num_filters = num_filters

        if poly_filter:
            self.mod = nn.ModuleList([nn.ModuleList([ ExpPolyFilter(max_poly_order = kwargs["max_poly_order"]) for _ in range(num_filters)]) for _ in range(input_dim)])
        elif scattering_filter:
            self.mod = nn.ModuleList([nn.ModuleList([ ScatteringFilter(degree) for degree in range(num_filters)]) for _ in range(input_dim)])
        else:
            self.mod = nn.ModuleList([nn.ModuleList([ nn.Sequential(nn.Linear(1,10),nn.ReLU(),nn.Linear(10,1)) for _ in range(num_filters)]) for _ in range(input_dim)])

    def forward(self,x):
        #return torch.cat([x.x[...,None] for _ in range(self.num_filters)], dim = -1) # bypass test
        filtered_outputs = []
        L = torch.sparse.FloatTensor(x.L_i, x.L_v, torch.Size(x.L_shape))
        for k in range(self.num_filters):
            try:
                coeffs = x.x.T @ L
            except:
                breakpoint()
            w = torch.stack([self.mod[i][k](x.node_attr_eig[:,None]) for i in range(self.input_dim)])[...,0]
            filtered_output = L @ (coeffs * w).T
            filtered_outputs.append(filtered_output)                
        filtered_tensor = torch.stack(filtered_outputs, -1) # Size = [N x input_dim x num_filters]
        return filtered_tensor



class ScatteringFilter(nn.Module):
    def __init__(self,degree):
        super().__init__()
        self.degree = degree
        if degree == 0:
            self.scales = [0,1]
        else:
            self.scales = [2**(degree-1), 2**degree]

    def forward(self,x):
        x_exp = torch.exp(-x)
        out = x_exp.pow(self.scales[0]) - x_exp.pow(self.scales[1])
        #poly_out = torch.sum(torch.stack([x_exp.pow(i)*self.alpha_k[i] for i in range(self.max_poly_order)]), dim = 0)
        return out




class ScatteringFilter_(nn.Module):
    def __init__(self,input_dim, num_filters, K, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.num_filters = num_filters

        self.g = lambda x: torch.exp(-x)
        self.exponents = 2**torch.arange(num_filters+1)
    
    def forward(self,x):
        L = torch.sparse.FloatTensor(x.L_i, x.L_v, torch.Size(x.L_shape))
        coeffs = x.x.T @ L # input_dim x N
        w = self.g(x.node_attr_eig)[:,None].pow(self.exponents[None,:]) # N x num_filters
        w_wav = w[:,:-1] - w[:,1:] # N x num_filters-1
        g_eig = (coeffs[...,None] * w_wav[None,...]).permute(1,0,2) # N x input_dim x num_filters-1
        filtered_output = L @ (g_eig.reshape(-1,self.input_dim*(self.num_filters)))
        filtered_output = filtered_output.reshape(-1,self.input_dim,self.num_filters) # N x input_dim x num_filters -1 
        return filtered_output
        
class tensor_layer(nn.Module):
    def __init__(self, input_dim : int,
                  output_dims: list):
        super().__init__()
        assert len(output_dims)<=2
        self.input_dim = input_dim
        self.output_dims = output_dims

        self.tensor = nn.Parameter(torch.randn([input_dim]+output_dims))

    def forward(self, x):
        x = torch.randn([5,2,3])
        y = torch.randn([2,3,4])
        z = torch.einsum('bnd,ndk->bnk', x, y) 
        return torch.einsum('bnd,ndk->bnk', x, self.tensor)
