import torch
import numpy as np
class SimplicialWaveletTransform():
    def __init__(self, adj, ro, threshold, device):
        self.adj = adj
        self.device = device
        self.X = ro
        self.indices = []
        self.threshold = threshold
        self.B = [None, self.compute_B1(), self.compute_B2()]
        X = self.calculate_simplex_features()
        self.X = [ro, X[0], X[1]]
        del(X)
        # self.ll, self.ul = self._get_laplacians()
        self.P_B, self.P_L, self.P_U = self._get_transition_matrix()
        
    def compute_B1(self):
        n = self.adj.shape[0]
        i, j = torch.triu_indices(n, n, offset=1).to(self.device) 
        weights = self.adj[i, j]          

        non_zero_mask = weights > 0
        i, j = i[non_zero_mask], j[non_zero_mask]
        weights = weights[non_zero_mask]
        num_edges = len(weights)  
        
        index = {}
        for k in range(n):
            index[frozenset([k])] = k
        self.indices.append(index)
        index = {}
        self.edges = torch.stack((i,j)).T
        for k,v in enumerate(self.edges.tolist()):
            index[frozenset(v)] = k
        self.indices.append(index)
            
        B1 = torch.zeros((n, num_edges), device=self.adj.device)
        B1[i, torch.arange(num_edges)] = weights 
        B1[j, torch.arange(num_edges)] = weights
        return B1

    def compute_B2(self):
        n = self.adj.shape[0]

        i, j = torch.triu_indices(n, n, offset=1).to(self.device)   
        edge_weights = self.adj[i, j]    
        non_zero_mask = edge_weights > 0
        i, j = i[non_zero_mask], j[non_zero_mask]
        edge_weights = edge_weights[non_zero_mask]
        num_edges = len(edge_weights)

        potential_triangles = torch.combinations(torch.arange(n), r=3).to(self.device) 

        i_t, j_t, k_t = potential_triangles.T

        valid_triangles_mask = (
            (self.adj[i_t, j_t] > 2*self.threshold) &
            (self.adj[j_t, k_t] > 2*self.threshold) &
            (self.adj[i_t, k_t] > 2*self.threshold)
        ) 
        
        self.triangles = potential_triangles[valid_triangles_mask].cpu().numpy()
        if(len(self.triangles)>250):
            self.triangles = self.triangles[torch.randint(0, len(self.triangles), (250,))]
        num_triangles = self.triangles.shape[0]
        index = {}
        for k,v in enumerate(self.triangles.tolist()):
            index[frozenset(v)] = k
        self.indices.append(index)
        
        B2 = torch.zeros((num_edges, num_triangles), device=self.adj.device)
        
        idx = np.arange(1, 3) - np.tri(3, 2, k=-1, dtype=bool)
        for m,j in enumerate(self.triangles):
            for k in idx:
                B2[self.indices[1][frozenset(j[k])], self.indices[2][frozenset(j)]] = edge_weights[self.indices[1][frozenset(j[k])]]
        return B2
    
    def calculate_simplex_features(self):
        X1 = self.X[self.edges].mean(1)
        X2 = self.X[self.triangles].mean(1)
        return [X1, X2]
    
    def _get_laplacians(self):
        lower_laplacians = [None]*len(self.B)
        upper_laplacians = [None]*len(self.B)
        for i in range(1,len(self.B)):
            lower_laplacians[i] = self.B[i].T@self.B[i]
        for i in range(0,len(self.B)-1):
            upper_laplacians[i] = self.B[i+1]@self.B[i+1].T
        return lower_laplacians, upper_laplacians
    
    def _get_transition_matrix(self):
        P_B = [None]*len(self.B)
        P_U = [None]*len(self.B)
        P_L = [None]*len(self.B)
        for i in range(len(self.B)):
            if(self.B[i] is not None):
                P_B[i] = (torch.linalg.inv(torch.diag(self.B[i].sum(axis=1)) + torch.eye(self.B[i].shape[0]).to(self.device))@self.B[i]).to(self.device)
        for i in range(1, len(self.B)):
            ul = self.B[i].T@self.B[i]             
            
            P_U[i] = (ul@torch.linalg.inv(torch.diag(ul.sum(axis=1)) + torch.eye(ul.shape[0]).to(self.device))).to(self.device)
        for i in range(0,len(self.B)-1):
            ll = self.B[i+1]@self.B[i+1].T
            P_U[i] = (ll@torch.linalg.inv(torch.diag(ll.sum(axis=1)) + torch.eye(ll.shape[0]).to(self.device))).to(self.device)
        return P_B, P_L, P_U

    def message_passing(self, X, include_boundary):
        neighbors = []
        aggregate = []
        for k in range(len(X)):
            X_l = torch.zeros(X[k].shape).to(self.device)
            X_u = torch.zeros(X[k].shape).to(self.device)
            X_b = torch.zeros(X[k].shape).to(self.device)
            X_c = torch.zeros(X[k].shape).to(self.device)
            if(self.P_U[k] is not None):
                X_u = self.P_U[k]@X[k]
            if(self.P_L[k] is not None):
                X_l = self.P_L[k]@X[k]
            if include_boundary:
                if(k<len(X)-1):
                    if(self.B[k] is not None):
                        X_b = self.P_B[k+1]@X[k+1]
                if(self.B[k] is not None):
                    X_c = self.P_B[k].T@X[k-1]
            neighbors.append([X_b, X_c, X_l, X_u])
            aggregate.append(X[k]/5 + X_b + X_c + X_l + X_u)
        return neighbors, aggregate

    def calculate_Z(self, J, include_boundary):
        Z_agg = []
        Z_neigh = []
        for i in range(J):
            if(i==0):
                neigh, agg = self.message_passing(self.X, include_boundary)
                Z_agg.append(agg)
                Z_neigh.append(neigh)
            else:
                neigh, agg = self.message_passing(Z_agg[-1], include_boundary)
                Z_agg.append(agg)
                Z_neigh.append(neigh)
        return Z_agg, Z_neigh

    def scattering(self, X, Z_neigh, index, J):
        psi = []
        for i in index:
            p = []
            for j in range(J+1):
                out = []
                if(j==0):
                    for k in range(len(X)):
                        out.append(torch.zeros_like(X[k]))
                elif(j==J):
                    for k in range(len(X)):
                        out.append(torch.zeros_like(X[k]))
                else:
                    for k in range(len(X)):
                        out.append(torch.abs(Z_neigh[j-1][k][index[i]] - Z_neigh[j][k][index[i]]))
                p.append(out)
            psi.append(p)
        return psi

    def agg(self, psi, X, index, J):
        Psi = []
        for j in range(J):
            psi_j = []
            for k in range(len(X)):
                psi_j.append((X[k]+psi[index['B']][j][k]+psi[index['C']][j][k]+psi[index['L']][j][k]+psi[index['U']][j][k])/5)
            Psi.append(psi_j)
        return Psi

    def calculate_wavelet_coeff(self, J, output_size = None, include_boundary = True):
        index = {'B':0, 'C':1, 'L':2, 'U':3}
        Z_agg, Z_neigh = self.calculate_Z(J, include_boundary)
        psi = self.scattering(self.X, Z_neigh, index, J)
        Psi_j = self.agg(psi, self.X, index, J)
        PSI = []
        for PsiX in Psi_j:
            PSI.append(torch.cat([k.sum(0) for k in PsiX], dim=0))
        return torch.cat(PSI)