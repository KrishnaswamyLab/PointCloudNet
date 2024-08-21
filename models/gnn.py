from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        for i in range(num_layers-2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x, edge_index, batch):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x, edge_index)
            x = x.relu()
        x = global_mean_pool(x, batch)  
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layers[-1](x)
        
        return x