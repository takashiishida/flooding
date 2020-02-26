import torch
import torchvision
import numpy as np
from copy import deepcopy


class SpiralDataNumpy:
    def __init__(self, label_noise=0.1, noise_level=1, n_pos_tr=50, n_neg_tr=50, n_pos_va=10, n_neg_va=10, n_pos_te=10000, n_neg_te=10000):
        self.label_noise = label_noise
        self.noise_level = noise_level
        self.n_pos_tr, self.n_neg_tr = n_pos_tr, n_neg_tr
        self.n_pos_va, self.n_neg_va = n_pos_va, n_neg_va
        self.n_pos_te, self.n_neg_te = n_pos_te, n_neg_te

    def sub_makeData(self, n_pos, n_neg, label_noise, noise_level):        
        a1, a2 = np.linspace(0, 4*np.pi, n_pos), np.linspace(0, 4*np.pi, n_neg)
        n = n_pos + n_neg
        y, x = np.zeros(shape=(n,)), np.zeros(shape=(n, 2))

        y[:n_pos], y[n_pos:] = 1, -1        
        x[y == 1, 0] = a1 * np.cos(a1) + noise_level*np.random.randn(n_pos)
        x[y == 1, 1] = a1 * np.sin(a1) + noise_level*np.random.randn(n_pos)
        x[y == -1, 0] = (a2 + np.pi) * np.cos(a2) + noise_level*np.random.randn(n_neg)
        x[y == -1, 1] = (a2 + np.pi) * np.sin(a2) + noise_level*np.random.randn(n_neg)        
        
        """
        a_p, a_n = np.linspace(0, 4*np.pi, n_pos), np.linspace(0, 4*np.pi, n_neg)    
        x_p = np.array([a_p * np.cos(a_p), (a_p + np.pi) * np.cos(a_p)]) + noise_level*np.random.randn(2, n_pos)
        x_n = np.array([a_n * np.sin(a_n), (a_n + np.pi) * np.sin(a_n)]) + noise_level*np.random.randn(2, n_neg)
        x = np.r_[x_p.T, x_n.T]
        y = np.r_[-np.ones(n_pos), np.ones(n_neg)]
        """
        
        y_clean = deepcopy(y)
        if label_noise > 0:
            idx = np.random.permutation(y.shape[0])[:int(y.shape[0]*label_noise)]
            y[idx] *= -1
        y, y_clean = (y+1)//2, (y_clean+1)//2
        r_dummy = np.ones((x.shape[0], 2))
        return x, r_dummy, y, y_clean
        
    def makeData(self):
        x_tr, r_tr, y_tr, y_tr_clean = self.sub_makeData(self.n_pos_tr, self.n_neg_tr, self.label_noise, self.noise_level)
        x_va, r_va, y_va, y_va_clean = self.sub_makeData(self.n_pos_va, self.n_neg_va, self.label_noise, self.noise_level)
        x_te, r_te, y_te, y_te_clean = self.sub_makeData(self.n_pos_te, self.n_neg_te, self.label_noise, self.noise_level)        
        
        return x_tr.astype('float32'), r_tr.astype('float32'), y_tr.astype('int64'), y_tr_clean.astype('int64'), \
               x_va.astype('float32'), r_va.astype('float32'), y_va.astype('int64'), y_va_clean.astype('int64'), \
               x_te.astype('float32'), r_te.astype('float32'), y_te.astype('int64'), y_te_clean.astype('int64')