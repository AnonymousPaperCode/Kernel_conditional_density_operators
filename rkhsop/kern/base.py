#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np, scipy as sp, scipy.stats as stats

from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
import distributions as dist

from scipy.spatial.distance import squareform, pdist, cdist

def median_heuristic(data, distance, per_dimension = True):
    if isinstance(distance, str):
        dist_fn = lambda x: pdist(x, distance)
    else:
        dist_fn = distance
    if per_dimension is False:
        return np.median(dist_fn(data))
    else:
        def single_dim_heuristic(data_dim):
            return median_heuristic(data_dim[:, None], dist_fn, per_dimension = False)
        return np.apply_along_axis(single_dim_heuristic, 0, data)

class Kernel(object):
    def __call__(self, *args, **kwargs):
        return self.gram(*args, **kwargs)

    def gram(self, X, Y = None, diag = False):
        """compute the gram matrix, i.e. the kernel evaluated at every element of X paired with each element of Y"""
        raise NotImplementedError()

    def get_params(self):
        # get unconstrained parameters
        assert()

    def set_params(self, params):
        # set unconstrained parameters, possibly transform them
        assert()

    def rkhsel_gram(self, X, Y = None, logsp = False):
        """
        X - axis 0 contains observations (of sample sets), axis 1 is input dimension, axis 2 are different points per observation (the samples of a sample set)
        """

        assert(not logsp)
        if Y is not None:
            assert(len(Y.shape) == 2)
        if len(X.shape) == 2:
            return self.gram(X, Y)
        assert(len(X.shape) == 3)
        
        X_resh = np.concatenate(np.swapaxes(X,0,2), axis=1).T #np.swapaxes(X, 1,2).reshape(-1, X.shape[1])
        if Y is None:
            # compute the full gram matrix
            G = self.gram(X_resh)
            # sum up the blockmatrices of shape (X.shape[2], X.shape[2]) that make up G
            G = np.mean(np.split(np.mean(np.split(G, X.shape[2], 1), 0), X.shape[2]), 0)

            # return the matrix of RKHS inner products of the mean embeding objects
            return G
        else:
            return np.mean(np.split(self.gram(X_resh, Y), X.shape[2]), 0)


class FeatMapKernel(Kernel):
    def __init__(self, feat_map):
        self.features = feat_map

    def features_mean(self, samps):
        return self.features(samps).mean(0)

    def gram(self, X, Y = None, diag = False):
        f_X = self.features(X)
        if Y is None:
            f_Y = f_X
        else:
            f_Y = self.features(Y)
        if diag:
            return np.sum(f_X * f_Y, 1)
        else:
            return f_X.dot(f_Y.T)


class LinearKernel(FeatMapKernel):
    def __init__(self):
        FeatMapKernel.__init__(self, lambda x: x)
        
def int_softplus(x):
    if x >= 34:
        # in this case, limitations in floating-point
        # precision result in log(exp(y) - 1) == y
        return x
    elif x <= -37:
        # this also results from precision limits
        return 10**-8
    else:
        return log(1 + exp(x))
        
softplus = np.vectorize(int_softplus)

def int_invsoftplus(y):
    y = float(y)
    if y >= 34:
        # in this case, limitations in floating-point
        # precision result in log(exp(y) - 1) == y
        return y 
    elif y < 0:
        raise ValueError("Function defined only for y >= 0")
    elif y == 0:
        #perturb input so as not to return -inf
        y += stats.gamma(100,scale=0.00001).rvs()
    return log(exp(y) - 1)
invsoftplus = np.vectorize(int_invsoftplus)

class GaussianKernel(Kernel):
    def __init__(self, sigma, diffable = False):
        self.set_params(invsoftplus(sigma))
        self.diffable = diffable

    def get_params(self):
        return self.params

    def get_double_var_kern(self):
        return GaussianKernel(np.sqrt(2) * self._sd)

    def set_params(self, params):
        self.params = np.atleast_1d(params).flatten()[0]
        self.__set_standard_dev(softplus(self.params))

    def __set_standard_dev(self, sd):
        assert(np.all(sd > 0))
        self._sd = sd
        self._const_factor = -0.5 / sd**2
        self._normalization = (sqrt(2*np.pi)*sd)
        self._log_norm = log(self._normalization)

    def get_var(self):
        return self._sd**2

    def gram(self, X, Y=None, diag = False, logsp = False):
        assert(len(np.shape(X))==2)

        # if X=Y, use more efficient pdist call which exploits symmetry

        if diag:
            if Y is None:
                sq_dists = np.zeros(X.shape[0])
            else:
                assert(X.shape == Y.shape)
                sq_dists = np.sum((X - Y)**2, 1)
        else:
            if not self.diffable:
                if Y is None:
                    sq_dists = squareform(pdist(X, 'sqeuclidean'))
                else:
                    assert(len(Y.shape) == 2)
                    assert(X.shape[1] == Y.shape[1])
                    sq_dists = cdist(X, Y, 'sqeuclidean')
            else:
                assert(len(np.shape(Y))==2)
                assert(np.shape(X)[1]==np.shape(Y)[1])
                sq_dists = ((np.tile(X,(Y.shape[0], 1)) - np.repeat(Y, X.shape[0], 0))**2).sum(-1).reshape(Y.shape[0], X.shape[0]).T

        rval = self._const_factor* sq_dists - self._log_norm * np.shape(X)[1]
        if not logsp:
            return exp(rval)
        return rval

    def rvs(self, nrows, ncols):
        return np.random.randn(nrows, ncols) * self._sd

class LaplaceKernel(Kernel):
    def __init__(self, sigma, diffable = False):
        self.set_params(log(exp(sigma) - 1))
        self.diffable = diffable

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = np.atleast_1d(params).flatten()[0]
        self.__set_standard_dev(log(exp(self.params) + 1))

    def __set_standard_dev(self, sd):
        self._sd = sd
        self._scale = sd
        self._const_factor = -1./self._scale
        self._normalization = 2 * self._scale
        self._log_norm = log(self._normalization)

    def get_var(self):
        return 2 * self._scale**2

    def gram(self, X, Y=None, diag = False, logsp = False):
        assert(len(np.shape(X))==2)

        # if X=Y, use more efficient pdist call which exploits symmetry

        if diag:
            if Y is None:
                dists = np.zeros(X.shape[0])
            else:
                assert(X.shape == Y.shape)
                dists = np.sum((X - Y)**2, 1)
        else:
            if not self.diffable:
                if Y is None:
                    dists = squareform(pdist(X, 'cityblock'))
                else:
                    assert(len(Y.shape) == 2)
                    assert(X.shape[1] == Y.shape[1])
                    dists = cdist(X, Y, 'cityblock')
            else:
                assert(len(np.shape(Y))==2)
                assert(np.shape(X)[1]==np.shape(Y)[1])
                assert("This is not tested!")
                dists = (np.abs(np.tile(X,(Y.shape[0], 1)) - np.repeat(Y, X.shape[0], 0))).sum(-1).reshape(Y.shape[0], X.shape[0]).T

        rval = self._const_factor * dists - self._log_norm * np.shape(X)[1]
        if not logsp:
            return exp(rval)
        return rval

    def rvs(self, nrows, ncols):
        return stats.laplace.rvs(scale=self._scale, size = (nrows, ncols))



class StudentKernel(Kernel):
    def __init__(self, s2, df, diffable = False):
        self.set_params(log(exp(np.array([s2,df])) - 1))
        self.diffable = diffable


    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params
        self.dens = dist.mvt(0, log(exp(params[0]) + 1), log(exp(params[1]) + 1))

    def gram(self, X, Y=None, diag = False, logsp = False):
        assert(len(np.shape(X))==2)

        # if X=Y, use more efficient pdist call which exploits symmetry

        if diag:
            if Y is None:
                sq_dists = np.zeros(X.shape[0])
            else:
                assert(X.shape == Y.shape)
                sq_dists = np.sum((X - Y)**2, 1)
        else:
            if not self.diffable:
                if Y is None:
                    sq_dists = squareform(pdist(X, 'sqeuclidean'))
                else:
                    assert(len(Y.shape) == 2)
                    assert(X.shape[1] == Y.shape[1])
                    sq_dists = cdist(X, Y, 'sqeuclidean')
            else:
                assert(len(np.shape(Y))==2)
                assert(np.shape(X)[1]==np.shape(Y)[1])
                sq_dists = ((np.tile(X,(Y.shape[0], 1)) - np.repeat(Y, X.shape[0], 0))**2).sum(-1).reshape(Y.shape[0], X.shape[0]).T
        dists = np.sqrt(sq_dists)
        rval = self.dens.logpdf(dists.flatten()).reshape(dists.shape)
        if not logsp:
            return rval
        else:
            return exp(rval)

class SplitDimsKernel(Kernel):
    def __init__(self, intervals, kernels, operation = "*", weights = None):
        assert(len(intervals) - 1 == len(kernels))
        self.intervals = intervals
        self.kernels = kernels
        if operation == "*":
            self.weights = np.ones(len(kernels))
            self.op = lambda x: np.prod(x, 0)
            self.log_op = lambda x: np.sum(x, 0)
            if weights is None:
                self.weights = np.ones(len(kernels))
            else:
                self.weights = weights
        else:
            assert(operation == '+')
            self.op = lambda x: np.sum(x, 0)
            self.log_op = lambda x: logsumexp(x, 0)
            if weights is None:
                self.weights = np.ones(len(kernels))
            else:
                self.weights = weights
        
    def gram(self, X, Y=None, diag = False, logsp = False):
        assert(not logsp)
        
        split_X = [X[:, self.intervals[i]:self.intervals[i + 1]] for i in range(len(self.kernels))]
        if Y is None:
            split_Y = [None] * len(self.kernels)
        else:
            split_Y = [Y[:, self.intervals[i]:self.intervals[i + 1]] for i in range(len(self.kernels))]
        sub_grams = np.array([self.kernels[i].gram(split_X[i], split_Y[i], diag = diag) * self.weights[i]  for i in range(len(self.kernels))])
        return self.op(sub_grams)

def test_SplitDimsKernel():
    (intervals, kernels) = ([0, 2, 5], [GaussianKernel(0.1), GaussianKernel(1)])
    X = np.arange(15).reshape((3,5))
    Y = (X + 3)[:-1,:]
    for op in "+", "*":
        k = SplitDimsKernel(intervals, kernels, op)
        assert(k.gram(X, Y).shape == (len(X), len(Y)))
        assert(k.gram(X).shape == (len(X), len(X)))
        assert(k.gram(X, diag = True).shape == (len(X),))
                

class SKlKernel(Kernel):
    def __init__(self, sklearn_kernel):
        self.skl = sklearn_kernel

    def gram(self, X, Y=None, diag = False, logsp = False):
        if diag:
            assert(Y is None)
            rval = self.skl.diag(X)
        else:
            rval = self.skl(X, Y)
        if logsp:
            rval = log(rval)
        return rval
