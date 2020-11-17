import torch
import torchvision
import numpy as np
from copy import deepcopy


class GaussianDataTorch:
    def __init__(self, mu_pos, mu_neg, n_tr=10, n_te=1000, d=2):
        self.n_tr, self.n_te = n_tr, n_te
        self.d = d
        self.mu_pos, self.mu_neg = mu_pos, mu_neg

    def makeData(self):
        pos_dist = torch.distributions.MultivariateNormal(torch.ones(self.d)*self.mu_pos, torch.eye(self.d))
        neg_dist = torch.distributions.MultivariateNormal(torch.ones(self.d)*self.mu_neg, torch.eye(self.d))
        pos_tr = pos_dist.sample((self.n_tr,))
        neg_tr = neg_dist.sample((self.n_tr,))
        x = torch.cat((pos_tr,neg_tr))
        y = torch.cat((torch.zeros(self.n_tr), torch.ones(self.n_tr)))
        return x, y

class GaussianDataNumpy:
    def __init__(self, mu_pos, mu_neg, cov_pos=np.identity(10), cov_neg=np.identity(10), n_pos_tr=50, n_neg_tr=50, 
                 n_pos_va=10000, n_neg_va=10000, n_pos_te=10000, n_neg_te=10000, label_noise=0):
        '''The ratio of n_pos/n_neg and n_pos_te/n_neg_te should be the same.
        Using the same number of samples for test and validation; only specify test information in args.'''
        self.n_pos_tr, self.n_neg_tr = n_pos_tr, n_neg_tr
        self.n_pos_va, self.n_neg_va = n_pos_va, n_neg_va
        self.n_pos_te, self.n_neg_te = n_pos_te, n_neg_te
        
        self.d = len(mu_pos)
        self.mu_pos, self.mu_neg = mu_pos, mu_neg
        self.cov_pos, self.cov_neg = cov_pos, cov_neg
        self.pos_prior = self.n_pos_tr / (self.n_pos_tr + self.n_neg_tr)
        self.label_noise = label_noise

    def sub_makeData(self, n_p, n_n, label_noise):
        x_p = np.random.multivariate_normal(self.mu_pos, self.cov_pos, n_p)
        x_n = np.random.multivariate_normal(self.mu_neg, self.cov_neg, n_n)
        x = np.r_[x_p, x_n]
        r = self.getR(n_p, n_n, x)
        y = np.r_[-np.ones(n_p), np.ones(n_n)]  
        y_clean = deepcopy(y)           
        if label_noise > 0:
            n = y.shape[0]
            idx = np.random.permutation(n)[:int(n*label_noise)]
            y[idx] *= -1
        y, y_clean = (y+1)//2, (y_clean+1)//2
        return x, r, y, y_clean
        
    def makeData(self):
        x_tr, r_tr, y_tr, y_tr_clean = self.sub_makeData(self.n_pos_tr, self.n_neg_tr, self.label_noise)
        x_va, r_va, y_va, y_va_clean = self.sub_makeData(self.n_pos_va, self.n_neg_va, self.label_noise)
        x_te, r_te, y_te, y_te_clean = self.sub_makeData(self.n_pos_te, self.n_neg_te, self.label_noise)        
        
        return x_tr.astype('float32'), r_tr.astype('float32'), y_tr.astype('int64'), y_tr_clean.astype('int64'), \
               x_va.astype('float32'), r_va.astype('float32'), y_va.astype('int64'), y_va_clean.astype('int64'), \
               x_te.astype('float32'), r_te.astype('float32'), y_te.astype('int64'), y_te_clean.astype('int64')        
    

    def getPositivePosterior(self, x):
        """Returns the positive posterior p(y=+1|x)."""
        conditional_pos = np.exp(-0.5 * (x - self.mu_pos).T.dot(np.linalg.inv(self.cov_pos)).dot(x - self.mu_pos)) / np.sqrt(np.linalg.det(self.cov_pos)*(2 * np.pi)**x.shape[0])
        conditional_neg = np.exp(-0.5 * (x - self.mu_neg).T.dot(np.linalg.inv(self.cov_neg)).dot(x - self.mu_neg)) / np.sqrt(np.linalg.det(self.cov_neg)*(2 * np.pi)**x.shape[0])
        marginal_dist = self.pos_prior * conditional_pos + (1 - self.pos_prior) * conditional_neg
        posterior_pos = conditional_pos * self.pos_prior / marginal_dist
        return posterior_pos

    def getR(self, n_pos, n_neg, x):
        """calculating the exact positive-confidence values: r. x should be the input dataset"""
        r = np.zeros((n_pos+n_neg, 2)) # n times k
        for i in range(len(r)):
            pos_conf = self.getPositivePosterior(x[i,:])
            r[i,0], r[i,1] = pos_conf, 1-pos_conf
        return r

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data, label, conf, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])):
        self.transform = transform
        self.data = data
        self.conf = conf
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.transform(self.data)[0][idx]
        out_label = self.label[idx]
        out_conf = self.conf[idx]
        return out_data, out_label, out_conf
