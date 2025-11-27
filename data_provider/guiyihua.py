import numpy as np
import math
import random
import torch

import numpy as np


class SimilarityMatrixUpdater:
    def __init__(self, args,sm1, sm2, k1):

        self.sm1 = sm1
        self.sm2 = sm2
        self.k = k1
        self.w1 = args.w1


    # W is the matrix which needs to be normalized
    def new_normalization(self, w):
        m = w.shape[0]
        p = np.zeros([m, m])
        for i in range(m):
            for j in range(m):
                if i == j:
                    p[i][j] = 1 / 2
                elif np.sum(w[i, :]) - w[i, i] > 0:
                    p[i][j] = w[i, j] / (2 * (np.sum(w[i, :]) - w[i, i]))
        return p

    # Get the KNN kernel, k is the number of first nearest neighbors
    def KNN_kernel(self, S, k):
        n = S.shape[0]
        S_knn = np.zeros([n, n])
        for i in range(n):
            sort_index = np.argsort(S[i, :])
            for j in sort_index[n - k:n]:
                if np.sum(S[i, sort_index[n - k:n]]) > 0:
                    S_knn[i][j] = S[i][j] / (np.sum(S[i, sort_index[n - k:n]]))
        return S_knn

    def disease_updating(self, S1, S2, P1, P2):
        it = 0
        P = (P1 + P2) / 2
        dif = 1
        while dif > 0.0000001:
            it += 1
            P111 = np.dot(np.dot(S1, P2), S1.T)
            P111 = self.new_normalization(P111)
            P222 = np.dot(np.dot(S2, P1), S2.T)
            P222 = self.new_normalization(P222)
            P1 = P111
            P2 = P222
            P_New = (1-self.w1) * P1 + self.w1 * P2
            dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
            P = P_New
        # print("Iteration number:", it)
        return P

    def get_syn_sim(self):

        if isinstance(self.sm1, torch.Tensor):
            self.sm1 = self.sm1.cpu().numpy()  #
        if isinstance(self.sm2, torch.Tensor):
            self.sm2 = self.sm2.cpu().numpy()  #

        d1 = self.new_normalization(self.sm1)
        d2 = self.new_normalization(self.sm2)


        Sd_1 = self.KNN_kernel(self.sm1, self.k)
        Sd_2 = self.KNN_kernel(self.sm2, self.k)


        Pd = self.disease_updating(Sd_1, Sd_2, d1, d2)


        Pd_final = (Pd + Pd.T) / 2

        return Pd_final

















