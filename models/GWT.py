from os import listdir
import numpy as np
import phate
import h5py
import scipy
import scipy.io as sio
from collections import Counter
from collections import defaultdict
from numpy import linalg as LA
import matplotlib.pyplot as plt
import networkx as nx
import tphate
import scprep


class GraphWaveletTransform():
    '''
    This class is used to generate graph wavelet transform features from a given adjacency matrix and node features.
    The graph wavelet transform is a method to generate features from a graph that are invariant to the graph's structure.'''

    def __init__(self, adj, ro):
        self.adj = adj
        self.ro = ro

    def lazy_random_walk(self):

        P_array = []
        d = self.adj.sum(0)
        P_t = self.adj/d
        P_t[np.isnan(P_t)] = 0
        P = 1/2*(np.identity(P_t.shape[0])+P_t)

        return P

    def graph_wavelet(self, P):

        psi = []
        for d1 in [1,2,4,8,16]:
            W_d1 = LA.matrix_power(P,d1) - LA.matrix_power(P,2*d1)
            psi.append(W_d1)

        return psi

    def zero_order_feature(self):

        F0 = np.matmul(LA.matrix_power(self.adj,16),self.ro)

        return F0

    def first_order_feature(self):

        P = self.lazy_random_walk(self.adj)
        W = self.graph_wavelet(P)
        u = np.abs(np.matmul(W,self.ro))

        F1 = np.matmul(LA.matrix_power(self.adj,16),np.abs(u))
        F1 = np.concatenate(F1,0)

        return F1

    def second_order_feature(self):

        P = self.lazy_random_walk(self.adj)
        W = self.graph_wavelet(P)
        u = np.abs(np.matmul(W,self.ro))

        u1 = np.einsum('ij,ajt ->ait',W[1],u[0:1])
        for i in range(2,len(W)):
            u1 = np.concatenate((u1,np.einsum('ij,ajt ->ait',W[i],u[0:i])),0)
        u1 = np.abs(u1)
        F2 = np.matmul(LA.matrix_power(self.adj,16),u1)
        F2 = np.concatenate(F2,0)

        return F2

    def generate_timepoint_feature(self):

        P = self.lazy_random_walk(self.adj)

        W = self.graph_wavelet(P)
        u = np.abs(np.matmul(W,self.ro))

        F0 = self.zero_order_feature(self.adj,self.ro)
        F1 = self.first_order_feature(self.adj,u)
        F2 = self.second_order_feature(self.adj,W,u)
        F = np.concatenate((F0,F1),axis=0)
        F = np.concatenate((F,F2),axis=0)

        return F


