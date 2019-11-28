import numpy as np
from scipy.stats import norm


__all__ =['Banana']


def logpdf(x, bananicity=0.03, V=100, compute_grad=False):
    if not compute_grad:
        return banana_log_pdf_theano(x, bananicity, V)
    else:
        return banana_log_pdf_grad_theano(x, bananicity, V)

def rvs(N, D, bananicity=0.03, V=100):
    X = np.random.randn(N, D)
    X[:, 0] = np.sqrt(V) * X[:, 0]
    X[:, 1:] = X[:, 1:] + bananicity * (X[:, :1] ** 2 - V)

    return X

def norm_of_emp_mean(X):
    return np.linalg.norm(np.mean(X, 0))

class Banana(object):
    def __init__(self, D, bananicity=0.03, V=100):
        self.bananicity = bananicity
        self.V = V
        self.D = D
        
    def rvs(self, n=1):
        return rvs(n, self.D, self.bananicity, self.V)
        
    def logpdf(self, X):
        transformed = X.copy()
        transformed[:, 1] = X[:, 1] - self.bananicity * ((X[:, 0] ** 2) - self.V)
        transformed[:, 0] = X[:, 0] / np.sqrt(self.V)
        return norm.logpdf(transformed).sum(1)
    