from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_mean_pool
import torch.nn as nn
import torch
from torch_geometric.nn import MessagePassing

class WeightedGCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(WeightedGCNLayer, self).__init__(aggr='add')  # Aggregation method (sum)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj):
        # x: Node feature matrix of shape [n, in_channels]
        # adj: Weighted adjacency matrix [n, n] (learnable with gradients)
        support = self.linear(x)  # Apply linear transformation
        out = torch.matmul(adj, support)  # Weighted sum of neighbors
        return out

class PointCloudGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_points):
        super(PointCloudGCN, self).__init__()
        # GCN layers
        self.gcn1 = WeightedGCNLayer(in_channels, hidden_channels)
        self.gcn2 = WeightedGCNLayer(hidden_channels, hidden_channels)
        self.gcn3 = WeightedGCNLayer(hidden_channels, out_channels)

    def forward(self, x, W):
        adj = torch.sigmoid(W)  # Optional: Ensure values are in [0, 1]

        # Forward pass through the GCN layers
        x = self.gcn1(x, adj)
        x = F.relu(x)
        x = self.gcn2(x, adj)
        x = F.relu(x)
        x = self.gcn3(x, adj)
        return x
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
    