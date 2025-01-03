"""
Pointnet++ pytorch implementation: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
"""

import os.path as osp
import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
#from torch_geometric.typing import WITH_TORCH_CLUSTER
#from torch_geometric.typing import OptTensor, torch_cluster
import torch_geometric.typing

#Takes in input the ratio, r and an MLP/neural network
#What is the ratio? : sampling ratio
#what is r?: radius/distance r
#pos: position matrix (coords of the points)
#fps(farthest point sampling): returns an index. iteratively samples the most distant point with regard to the rest points. takes in an input node feature matrix (pos), batch and sampling ratio
#radius: returns row col coords. find all the points within a distance of r to the "most distant point" as computed previously
#create an edge index with the rows and cols of the points obtained from the previous step
#Creates a convolution layer called pointnetconv and takes in an input of node features (x), pos (coords of the points), and edge index

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class Pointnet_plus(pl.LightningModule):
    def __init__(self, input_dim, pos_dim, num_classes,lr, **kwargs):
        super().__init__()
        self.lr = lr
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([input_dim+pos_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + pos_dim, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + pos_dim, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, num_classes], dropout=0.5, norm=None)
    
    def forward(self, data):
        sa0_out = (torch.ones((data.x.shape[0], 1)).to(data.x.device), data.x, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        # breakpoint()
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x).log_softmax(dim=-1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        y = batch.y
        logits = self(batch)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        # breakpoint()
        y = val_batch.y
        logits = self(val_batch)
        loss = F.nll_loss(logits, y)
        self.log("val_loss", loss)

        self.validation_step_outputs.append({'val_loss': loss,'y_hat': logits, 'y': y})

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])
        acc = torch.sum(y_hat.argmax(dim=1) == y).item() / (len(y) * 1.0)
        self.log('val_acc', acc)
        self.validation_step_outputs.clear()
        return acc
    
    def test_step(self, val_batch, batch_idx):
        #breakpoint()
        y = val_batch.y
        logits = self(val_batch)
        loss = F.nll_loss(logits, y)
        self.log("test_loss", loss)
        self.test_step_outputs.append({'test_loss': loss,'y_hat': logits, 'y': y})
        return loss

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])
        acc = torch.sum(y_hat.argmax(dim=1) == y).item() / (len(y) * 1.0)
        self.log('test_acc', acc)
        self.test_step_outputs.clear()
        return acc

def PointNetLoading(raw_dir, full, batch_size, device):
        raw_dir = raw_dir
        if raw_dir == "melanoma_data_full":
            if full:
                suffix = '_full'
            else:
                suffix = ''
            with open(osp.join(raw_dir, 'pc'+suffix+'.pkl'), 'rb') as handle:
                subsampled_pcs = pickle.load(handle)
                # subsampled_pcs = [torch.tensor(subsampled_pcs[i], dtype=torch.float).to(device) for i in range(len(subsampled_pcs))]
                subsampled_pcs = [torch.tensor(StandardScaler().fit_transform(subsampled_pcs[i]), dtype=torch.float).to(device) for i in range(len(subsampled_pcs))]
            labels = torch.LongTensor(np.load(osp.join(raw_dir, 'labels'+suffix+'.npy'))).to(device)
        elif raw_dir == "COVID_data":
            with open(osp.join(raw_dir, 'pc_covid.pkl'), 'rb') as handle:
                subsampled_pcs = pickle.load(handle)
                # subsampled_pcs = [torch.tensor(subsampled_pcs[i], dtype=torch.float).to(device) for i in range(len(subsampled_pcs))]
                subsampled_pcs = [torch.tensor(StandardScaler().fit_transform(subsampled_pcs[i]), dtype=torch.float).to(device) for i in range(len(subsampled_pcs))]
            with open(osp.join(raw_dir, 'patient_list_covid.pkl'), 'rb') as handle:
                subsampled_patient_ids = pickle.load(handle)
            labels = torch.LongTensor(np.load(osp.join(raw_dir, 'labels.npy'))).to(device)
        elif raw_dir == "pdo_data":
            with open(osp.join(raw_dir, 'pc_pdo_treatment.pkl'), 'rb') as handle:
                subsampled_pcs = pickle.load(handle)
                # subsampled_pcs = [torch.tensor(subsampled_pcs[i], dtype=torch.float).to(device) for i in range(len(subsampled_pcs))]
                subsampled_pcs = [torch.tensor(StandardScaler().fit_transform(subsampled_pcs[i]), dtype=torch.float).to(device) for i in range(len(subsampled_pcs))]
            le = LabelEncoder()
            labels = torch.LongTensor(le.fit_transform(np.load(osp.join(raw_dir, 'labels_pdo_treatment.npy')))).to(device)
        train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size = 0.2)
        val_idx, test_idx = train_test_split(test_idx, test_size = 0.5)
        graphs = [Data(x = subsampled_pcs[i], y = labels[i]) for i in range(len(labels))]
        train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader, subsampled_pcs[0].shape[1], len(torch.unique(labels))