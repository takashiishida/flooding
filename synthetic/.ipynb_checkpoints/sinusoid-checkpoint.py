import torch
import torchvision
import numpy as np
from copy import deepcopy


class SinusoidDataNumpy:
    def __init__(self, dim=2, label_noise=0.1, n_tr=50, n_va=10, n_te=20000):
        self.n_tr, self.n_va, self.n_te = n_tr, n_va, n_te
        self.dim = dim
        self.label_noise = label_noise

    def sub_makeData(self, n, label_noise):
        x = 2*np.random.randn(n, self.dim)
        u = np.zeros(shape=(1, self.dim))
        u[0, 0] = 1
        v = np.zeros(shape=(1, self.dim))
        v[0, 1] = 1
        y = np.sign(u.dot(x.T) + np.sin(v.dot(x.T))).astype(np.int32).reshape(n)        
        y_clean = deepcopy(y)
        if label_noise > 0:
            idx = np.random.permutation(n)[int(n*label_noise)]
            y[idx] *= -1
        y, y_clean = (y+1)//2, (y_clean+1)//2
        r_dummy = np.ones((n, 2))
        return x, r_dummy, y, y_clean
        
    def makeData(self):
        x_tr, r_tr, y_tr, y_tr_clean = self.sub_makeData(self.n_tr, self.label_noise)
        x_va, r_va, y_va, y_va_clean = self.sub_makeData(self.n_va, self.label_noise)
        x_te, r_te, y_te, y_te_clean = self.sub_makeData(self.n_te, self.label_noise)        
        
        return x_tr.astype('float32'), r_tr.astype('float32'), y_tr.astype('int64'), y_tr_clean.astype('int64'), \
               x_va.astype('float32'), r_va.astype('float32'), y_va.astype('int64'), y_va_clean.astype('int64'), \
               x_te.astype('float32'), r_te.astype('float32'), y_te.astype('int64'), y_te_clean.astype('int64')


class Sinusoid2dimDataNumpy:
    def __init__(self, label_noise=0.1, n_tr=50, n_va=10, n_te=20000):
        self.n_tr, self.n_va, self.n_te = n_tr, n_va, n_te        
        self.label_noise = label_noise

    def sub_makeData(self, n, label_noise):
        # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        s, t, u, v = np.random.randn(n), np.random.randn(n), np.random.randn(n), np.random.randn(n)
        norm = (s**2 + t**2 + u**2 + v**2)**(0.5)        
        x = 2*np.c_[u/norm, v/norm]        
        #w, _ = np.linalg.qr(np.random.randn(2, 2))
        #w *= 1.5
        w = 1.5*np.eye(2)
        y = np.sign(x.dot(w[:, 0]) + np.sin(x.dot(w[:, 1]))).astype(np.int32).reshape(n)        
        y_clean = deepcopy(y)
        if label_noise > 0:
            idx = np.random.permutation(n)[:int(n*label_noise)]
            y[idx] *= -1
        y, y_clean = (y+1)//2, (y_clean+1)//2    
        r_dummy = np.ones((n, 2))
        return x, r_dummy, y, y_clean
        
    def makeData(self):
        x_tr, r_tr, y_tr, y_tr_clean = self.sub_makeData(self.n_tr, self.label_noise)
        x_va, r_va, y_va, y_va_clean = self.sub_makeData(self.n_va, self.label_noise)
        x_te, r_te, y_te, y_te_clean = self.sub_makeData(self.n_te, self.label_noise)        
        
        return x_tr.astype('float32'), r_tr.astype('float32'), y_tr.astype('int64'), y_tr_clean.astype('int64'), \
               x_va.astype('float32'), r_va.astype('float32'), y_va.astype('int64'), y_va_clean.astype('int64'), \
               x_te.astype('float32'), r_te.astype('float32'), y_te.astype('int64'), y_te_clean.astype('int64')
    

    def view(self):
        x, _, y = self.sub_makeData(5000, 0.1)
        import matplotlib.pyplot as plt
        
        plt.plot(x[y == 0, 0], x[y == 0, 1], 'rx')
        plt.plot(x[y == 1, 0], x[y == 1, 1], 'bo')
        plt.show()
    