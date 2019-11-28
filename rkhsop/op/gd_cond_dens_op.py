#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import division, print_function, absolute_import

from numpy.random import permutation
import autograd.numpy as np
import autograd.scipy as sp

import autograd.scipy.stats as stats
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from autograd.numpy import exp, log, sqrt
from autograd.scipy.misc import logsumexp
from autograd import grad




import pylab as pl, matplotlib.pyplot as plt

import distributions as dist
import rkhsop.base as ro


class MatrKerIndDensityOp(ro.RKHSOperator):
    def __init__(self, inp, outp, kernel_input, kernel_output, num_induc, fit_matr = True, fit_kern = True, fit_induc = True, triang_matrix = False):
        rnd_sel = np.random.permutation(len(inp))[:num_induc]
        self.inp = inp[rnd_sel]
        self.outp = outp[rnd_sel]
        self.k_inp = kernel_input
        self.k_out = kernel_output
        self.num_induc = num_induc
        self.nparams = num_induc**2 + kernel_input.get_params().size + kernel_output.get_params().size
        self.logmatr = np.log(np.abs(np.random.randn(num_induc, num_induc)/20) + np.eye(num_induc))
        self.logmatr = self.logmatr - logsumexp(self.logmatr)
        (self.fit_matr, self.fit_kern, self.fit_induc) = (fit_matr, fit_kern, fit_induc)
        if triang_matrix:
            assert()
            self.matr_mask = np.triu(np.ones((num_induc, num_induc)) - np.infty)
        else:
            self.matr_mask = np.zeros((num_induc, num_induc))
        ro.RKHSOperator.__init__(self, self.inp, self.outp, kernel_input, kernel_output, True)

    
    def set_params(self, params):
        if self.fit_matr:
            self.logmatr = params[:self.num_induc**2].reshape(self.logmatr.shape) + self.matr_mask
            self.logmatr = self.logmatr - logsumexp(self.logmatr)
            params = params[self.num_induc**2:]
        if self.fit_kern:
            self.k_inp.set_params(exp(params[:self.k_inp.get_params().size]))
            params = params[self.k_inp.get_params().size:]
            self.k_out.set_params(exp(params[:self.k_out.get_params().size]))
            params = params[self.k_out.get_params().size:]
        if self.fit_induc:
            self.inp = params[:self.inp.size].reshape(self.inp.shape)
            params = params[self.inp.size:]
            self.outp = params[:self.outp.size].reshape(self.outp.shape)
        
            
    def get_params(self):      
        rval = []
        if self.fit_matr:
            rval.append(self.logmatr.flatten())
        if self.fit_kern:
            rval.extend([log(self.k_inp.get_params().flatten()),
                          log(self.k_out.get_params().flatten())])
        if self.fit_induc:
            rval.extend([self.inp.flatten(),
                          self.outp.flatten()])
        return np.hstack(rval)
    
    def neg_lhood_params(self, params, inp_data = None, out_data = None):
        self.set_params(params)
        return -self.lhood(inp_data, out_data, diag = True, logsp=True).sum()
        
    
    def fit(self, inp, outp, inducing_params = None):
        if self.fit_induc == False and self.fit_kern == False and self.fit_matr == False:
            print("Nothing to fit")
            return
        g_nll = grad(self.neg_lhood_params)
        x0 = self.get_params()
        options = {'disp':True,
                   'maxiter':80,
#                   'gtol':1e-2
                   }
        
        res = minimize(self.neg_lhood_params, x0, args=(inp, outp), jac = g_nll, method='L-BFGS-B', options=options, callback=lambda x:print(self.neg_lhood_params(x, inp, outp), np.linalg.norm(g_nll(x, inp, outp))))
        self.set_params(res['x'])