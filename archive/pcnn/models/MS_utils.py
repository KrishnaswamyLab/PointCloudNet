#scattering functions

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import seaborn as sns

import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

import scprep
import phate
import pickle
import graphtools
import scipy
import scipy.sparse
from scipy.spatial import distance_matrix

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

#################
'''
inputs:
- data (index = unique cell ID)
- metadata (data.index, patient ID)
**optional
- diff operator P
'''
#################


def pickle_save(obj, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_patient_dict(data):
    ''' 
    data should be pd dataframe where indices represent non-unique patient_id
    '''
    patient_data = {}
    for i, p in enumerate(np.unique(data.index)):
        patient_data[p] = data.loc[data.index == p, :]

    metadata = pd.DataFrame()
    metadata["patient_id"] = data.index
    num_ids = np.array(metadata.groupby('patient_id', sort=False).cumcount()+1)
    metadata.index = [str(data.index[i]) + "_" + str(num_ids[i]) for i in range(0, data.shape[0])]

    return metadata, patient_data


#graph constructions and diff_ops

#gene exp
def diff_op(data, k):
    #KNN-adaptive gaussian kernel 
    dist_matrix = sklearn.metrics.pairwise_distances(data)
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    K_distances = distances[:,k] 
    sigma_j = np.broadcast_to(K_distances, (dist_matrix.shape[0],dist_matrix.shape[1]))
    sigma_i = sigma_j.T
    W = (np.exp(-np.divide(dist_matrix**2,sigma_i**2)) + np.exp(-np.divide(dist_matrix**2,sigma_j**2)))*0.5
    #row norm D^{-1}W 
    D = np.diag(np.sum(W, axis= 0))
    P = np.matmul(np.linalg.inv(D),W) 
    return P

def diff_op_alpha(data, k, alpha):
    #KNN-adaptive kernel with alpha decay
    dist_matrix = sklearn.metrics.pairwise_distances(data)
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    K_distances = distances[:,k] 
    sigma_j = np.broadcast_to(K_distances, (dist_matrix.shape[0],dist_matrix.shape[1]))
    sigma_i = sigma_j.T
    W = (np.exp(-np.divide(dist_matrix**alpha,sigma_i**2)) + np.exp(-np.divide(dist_matrix**alpha,sigma_j**2)))*0.5
    #row norm D^{-1}W 
    D = np.diag(np.sum(W, axis= 0))
    P = np.matmul(np.linalg.inv(D),W) 
    return P

def diff_op_spatial(data, k):
    graph = graphtools.Graph(data, n_pca = None, knn=k, decay=40, verbose = False,random_state = 42) 
    P = graph.diff_op.toarray()
    return P


###################

def exp_centroids(data_global, global_phate, patient_data, patient_id, phate_clust_op, n_centroids=10):
    if phate_clust_op is None:
        try:
            phate_clust_op = phate.PHATE(n_components=2, knn=5, n_landmark= n_centroids, verbose = False)
            clust_phate = phate_clust_op.fit_transform(data)
            cluster_ids = phate_clust_op.graph.clusters #in gene expression space
            centroids = data.groupby(phate_clust_op.graph.clusters).mean()
            globals()['phate_clust_op'] = phate_clust_op
        except:
            print('no phate_clust_op available')
    cluster_ids = phate_clust_op.graph.clusters
    patient_diff_potentials = phate_clust_op.diff_potential[data_global.index == patient_id]
    #print(patient_diff_potentials.shape)
    centroid_idx = np.argmin(patient_diff_potentials, axis=0)
    centroids_phate = global_phate[data_global.index == patient_id][centroid_idx]
    centroids_gexp = pdata_exp.iloc[centroid_idx,:]
    
    signals = np.zeros([patient_data.shape[0], len(np.unique(centroid_idx))])
    for i, c in enumerate(centroid_idx):
        if i >= len(np.unique(centroid_idx)):
            signals[i,i] =1
        signals[c,i] = 1

    return signals, centroids_phate, cluster_ids




#making diracs for wavelet placement

#spatial
def spatial_centroids(data, vmin=0, vmax=1000, step= 200):
    if data.shape[1]>2:
        print('data should only have x y coordinates')
    a = np.arange(vmin, vmax+step, step)
    centroids = np.array(np.meshgrid(a, a)).T.reshape(-1, 2)
    ds = []
    for i, c in enumerate(centroids):
        #cell that is closest neighbor in data
        ds.append(np.linalg.norm(data - np.array(c), axis = 1).argmin())
    signals = np.zeros([data.shape[0], len(np.unique(ds))])
    for i, c in enumerate(ds):
        signals[c,i] = 1
    return signals


class geom_scattering(object):
    '''
    #geom scattering 
    #
    '''
    def __init__(self,
                 data,
                 P,
                 order,
                 scales,
                 q,
                 **args):
        self.data = data
        self.order = 2
        self.scales = scales
        self.q = 4
        self.P = P
        self.signals = args.get('signals')

        js = np.exp2(scales)
        js_ = np.exp2(np.array([s-1 for s in scales]))
        P_1 = np.power.outer(self.P, js)
        P_2 = np.power.outer(self.P, js_)
        psi = P_1 - P_2
        self.psi = np.transpose(psi, (2, 0, 1))

    def zeroth_order_transform(self):
        self.zeroth_order = np.matmul(np.identity(self.data.shape[0]), self.signals)
        return self.zeroth_order
        #zeroth_order= zeroth_order.reshape(-1,1)

    def first_order_transform(self):
        self.first_order = np.absolute(np.tensordot(self.psi,self.zeroth_order, axes=1))
        return self.first_order

    def second_order_transform(self):
        for i in range(0, self.psi.shape[0]):
            c = np.absolute(np.matmul(self.psi[i],self.first_order))
            if i==0:
                self.second_order = c
            else:
                self.second_order = np.vstack([self.second_order,c])
        return self.second_order

    def calculate_scattering(self):

        if self.signals is None:
            print('no signals')
        else:
            self.zeroth_order_transform()
            if self.order >=1:
                self.first_order_transform()
            if self.order ==2:
                self.second_order_transform()

        


def calculate_stats(S, axis= 0):
    ''' 
    statistical moments of scattering features
    '''
    mean = np.mean(S, axis =axis)
    variance = scipy.stats.variation(S, axis= axis)
    skew = scipy.stats.skew(S, axis= axis)
    kurtosis = scipy.stats.kurtosis(S, axis= axis)
    return mean, variance, skew, kurtosis 


