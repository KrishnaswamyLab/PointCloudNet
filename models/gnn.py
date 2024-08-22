from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_mean_pool
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
    
class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList([GINConv(nn.Linear(input_dim, hidden_dim), train_eps=True)])
        for i in range(num_layers-2):
            self.layers.append(GINConv(nn.Linear(hidden_dim, hidden_dim), train_eps=True))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x, edge_index, batch):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x, edge_index)
            x = x.relu()
        x = global_mean_pool(x, batch)  
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layers[-1](x)
        
        return x
    
    
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList([GATConv(input_dim, hidden_dim)])
        for i in range(num_layers-2):
            self.layers.append(GATConv(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x, edge_index, batch):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x, edge_index)
            x = x.relu()
        x = global_mean_pool(x, batch)  
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layers[-1](x)
        
        return x
    
class SAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        for i in range(num_layers-2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x, edge_index, batch):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x, edge_index)
            x = x.relu()
        x = global_mean_pool(x, batch)  
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layers[-1](x)
        
        return x
    