"""
PointTransformer pytorch implementation: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/point_transformer_classification.py
"""

import os.path as osp
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
import pytorch_lightning as pl
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    MLP,
    PointTransformerConv,
    fps,
    global_mean_pool,
    knn,
    knn_graph,
)
# from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.utils import scatter


class TransformerBlock(torch.nn.Module):
    def __init__(self, pos_dim, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([pos_dim, 64, out_channels], norm=None, plain_last=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None,
                           plain_last=False)

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x

class TransitionDown(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)
        # print(id_clusters.size(0))
        # Max pool onto each cluster the features from knn in points
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                        dim_size=id_clusters.size(0), reduce='max')

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch

class point_transformer(pl.LightningModule):
    def __init__(self, input_dim, pos_dim, num_classes,lr, dim_model=[32, 64, 128, 256, 512], k=16, **kwargs):
        super().__init__()
        self.k = k
        self.lr = lr
        self.validation_step_outputs = []
        self.test_step_outputs = []
        # dummy feature is created if there is none given
        in_channels = max(input_dim, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)

        self.transformer_input = TransformerBlock(pos_dim = pos_dim,
                                                  in_channels = dim_model[0],
                                                  out_channels = dim_model[0])
        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.k))

            self.transformers_down.append(
                TransformerBlock(pos_dim = pos_dim,
                                 in_channels=dim_model[i + 1],
                                 out_channels=dim_model[i + 1]))

        # class score computation
        self.mlp_output = MLP([dim_model[-1], 64, num_classes], norm=None)

    def forward(self, data, batch=None):
        pos = data.x
        batch = data.batch
        # add dummy features in case there is none
        x = torch.ones((data.x.shape[0], 1), device=data.x.get_device())

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)

        # Class score
        out = self.mlp_output(x)

        return F.log_softmax(out, dim=-1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        y = batch.y
        logits = self(batch)
        loss = F.mse_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # breakpoint()
        y = val_batch.y
        logits = self(val_batch)
        # loss = F.mse_loss(logits, y)*len(logits)
        # self.log("val_loss", loss)

        self.validation_step_outputs.append({'y_hat': logits, 'y': y})
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])
        # acc = torch.sum(y_hat.argmax(dim=1) == y).item() / (len(y) * 1.0)
        # self.log('val_acc', acc)
        # self.validation_step_outputs.clear()
        return F.mse_loss(y_hat, y)
    
    def test_step(self, val_batch, batch_idx):
        #breakpoint()
        y = val_batch.y
        logits = self(val_batch)
        # loss = F.mse_loss(logits, y)
        # self.log("test_loss", loss)
        self.test_step_outputs.append({'y_hat': logits, 'y': y})
        # return loss
    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])
        # acc = torch.sum(y_hat.argmax(dim=1) == y).item() / (len(y) * 1.0)
        # self.log('test_acc', acc)
        # self.test_step_outputs.clear()
        return F.mse_loss(y_hat, y)
