import torch


class GraphWaveletTransform():
    '''
    This class is used to generate graph wavelet transform features from a given adjacency matrix and node features.
    The graph wavelet transform is a method to generate features from a graph that are invariant to the graph's structure.'''

    def __init__(self, adj, ro, device):
        self.adj = adj.to(device)
        self.ro = ro.to(device)
        self.device = device
        d = self.adj.sum(0)
        P_t = self.adj/d
        P_t[torch.isnan(P_t)] = 0
        self.P = 1/2*(torch.eye(P_t.shape[0]).to(self.device)+P_t)
        self.psi = []
        for d1 in [1,2,4,8,16]:
            W_d1 = torch.matrix_power(self.P,d1) - torch.matrix_power(self.P,2*d1)
            self.psi.append(W_d1)

    def zero_order_feature(self):

        F0 = torch.matrix_power(self.adj,16)@self.ro

        return F0

    def first_order_feature(self):
        u = [torch.abs(self.psi[i]@self.ro) for i in range(len(self.psi))]
        # u.append(torch.matrix_power(self.adj,16)@self.ro)
        F1 = torch.cat(u,1)
        return F1, u

    def second_order_feature(self,u):
        u1 = torch.empty((self.ro.shape)).to(self.device)
        for j in range(len(self.psi)):
            for j_prime in range(0,j):
                u1 = torch.cat((u1, torch.abs(self.psi[j_prime]@u[j])), 1)
            # u1 = torch.cat((u1, torch.matrix_power(self.adj,16)@u[j]),1)
        return u1

    def generate_timepoint_feature(self):

        F0 = self.zero_order_feature()
        F1,u = self.first_order_feature()
        F2 = self.second_order_feature(u)
        F = torch.concatenate((F0,F1),axis=1)
        F = torch.concatenate((F,F2),axis=1)

        return F


