import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import numpy as np
import os
import pickle
# os.chdir('..')
# print(os.getcwd())
from models.graph_learning import PointCloudGraphEnsemble
from argparse import ArgumentParser


if __name__ == '__main__':
    # Define the parameters using parser args
    parser = ArgumentParser(description="Graph Ensemble Layer for Point Clouds")
    parser.add_argument('--raw_dir', type=str, default = 'data/melanoma/raw', help="Directory where the raw data is stored")
    parser.add_argument('--num_kernels', type=int, default=3, help="Number of kernels in the graph ensemble")
    parser.add_argument('--kernel_type', type=str, choices=['gaussian', 'alpha_decay'], default='gaussian', help="Type of kernel function")
    
    args = parser.parse_args()

    model = PointCloudGraphEnsemble(args.raw_dir, args.num_kernels, args.kernel_type)

    # Construct the graphs
    graphs = model.graph_construct()
    #Each graphs is a list of length 1620 where 1620 is 540 patient samples * 3 kernels
    # import pdb; pdb.set_trace()
    # Print and inspect the graphs
    for graph in graphs:
        print(f'Graph for patient ID: {graph.patient_id}')
        print(f'Number of nodes: {graph.x.size(0)}')
        print(f'Number of edges: {graph.edge_index.size(1)}')
        print(f'Edge attributes: {graph.edge_attr.size()}')
        print(f'Label: {graph.y.item()}')
        print('---')