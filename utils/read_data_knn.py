import os
import pickle
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import torch 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def read_data(raw_dir, num_neighbors, task):
    # if full:
    #     suffix = '_full'
    # else:
    #     suffix = ''
    # with open(os.path.join(raw_dir, 'pc'+suffix+'.pkl'), 'rb') as handle:
    #     subsampled_pcs = pickle.load(handle)
    # with open(os.path.join(raw_dir, 'patient_list'+suffix+'.pkl'), 'rb') as handle:
    #     subsampled_patient_ids = pickle.load(handle)
    # labels = np.load(os.path.join(raw_dir, 'labels'+suffix+'.npy'))
    # with open(os.path.join(raw_dir, 'pc_covid.pkl'), 'rb') as handle:
    #     subsampled_pcs = pickle.load(handle)
    # with open(os.path.join(raw_dir, 'patient_list_covid.pkl'), 'rb') as handle:
    #     subsampled_patient_ids = pickle.load(handle)
    # labels = np.load(os.path.join(raw_dir, 'labels.npy'))
    with open(os.path.join(raw_dir, 'pc_pdo_'+task+'.pkl'), 'rb') as handle:
        subsampled_pcs = pickle.load(handle)
    # labels = np.array([i.mean() for i in np.load(os.path.join(raw_dir, 'labels_pdo_'+task+'.npy'), allow_pickle=True)])

    labels = np.load(os.path.join(raw_dir, 'labels_pdo_'+task+'.npy'), allow_pickle=True)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    graphs = []
    
    num_labels = len(np.unique(labels))
    for i in range(len(subsampled_pcs)):
        x = StandardScaler().fit_transform(subsampled_pcs[0])
        knn = NearestNeighbors(n_neighbors = num_neighbors)
        knn.fit(subsampled_pcs[i])
        adj_list = knn.kneighbors()[1]
        edge_list = []
        for j in range(len(adj_list)):
            for k in adj_list[j]:
                edge_list.append([j,k])
        edge_list = torch.LongTensor(edge_list).T
        g = Data(x=torch.tensor(subsampled_pcs[i], dtype=torch.float), edge_index=edge_list)
        g.y = labels[i]
        graphs.append(g)

    train_idx, test_idx = train_test_split(np.arange(len(graphs)), test_size=0.2)
    return graphs, num_labels, train_idx, test_idx

def get_dataloaders(graphs, train_idx, test_idx):
    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=64, shuffle=True)
    test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=64, shuffle=False)
    
    # train_size = int(len(graphs)*0.8)
    # train_loader = DataLoader(graphs[:train_size], batch_size=64, shuffle=True)
    # test_loader = DataLoader(graphs[train_size:], batch_size=64, shuffle=False)

    return train_loader, test_loader