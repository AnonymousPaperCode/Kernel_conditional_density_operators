#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np, scipy as sp, scipy.stats as stats

from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from numpy import exp, log, sqrt
from scipy.misc import logsumexp

from rkhsop.op.base import RKHSOperator

class TransfOp(RKHSOperator):        
    def __init__(self, inp, outp, kern, regul : float = 0.00001, compute_eigd = False, logsp = False):
        RKHSOperator.__init__(self, inp, outp, kern, kern, logsp)
        self.kern = kern
        self.eigd = None
        
    def __flip_eigd__(self):
        self.eigd = (self.eigd[0][::-1],
                     self.eigd[1][:, ::-1])
        
    def compute_ef(self, new_inp, upto = None):
        if upto is None:
            upto = self.eigd[0].size
        return self.kern.gram(new_inp, self.inp) @ self.eigd[1][:,:upto]
        

class ePF(TransfOp):
    def __init__(self, inp, outp, kern, regul : float = 0.00001, compute_eigd = False, logsp = False):
        TransfOp.__init__(self, inp, outp, kern, regul, compute_eigd, logsp)
        self.matr = np.linalg.inv(kern.gram(inp) + regul * np.eye(inp.shape[0]))
        if compute_eigd:
            self.eigd = np.linalg.eigh(kern.gram(inp, outp) @ self.matr)
            self.eigd = (self.eigd[0], self.matr @ self.eigd[1])
            self.__flip_eigd__()
            
class kK(TransfOp):
    def __init__(self, inp, outp, kern, regul : float = 0.00001, compute_eigd = False, logsp = False):
        TransfOp.__init__(self, inp, outp, kern, regul, compute_eigd, logsp)
        self.matr = np.linalg.inv(kern.gram(inp) + regul * np.eye(inp.shape[0]))
        if compute_eigd:
            self.eigd = np.linalg.eigh(self.matr @ kern.gram(outp, inp))
            self.__flip_eigd__()
            
            


class kPF(TransfOp):
    def __init__(self, inp, outp, kern, regul : float = 0.00001, compute_eigd = False, logsp = False):
        TransfOp.__init__(self, inp, outp, kern, regul, compute_eigd, logsp)
        G_io = kern.gram(inp, outp)
        inv_G_io = np.linalg.inv(G_io + regul * np.eye(inp.shape[0]))
        G_o = kern.gram(outp)
        inv_G_i = np.linalg.inv(kern.gram(inp) + regul * np.eye(inp.shape[0]))
        self.matr = inv_G_io @ inv_G_io.T @ G_o
        if compute_eigd:
            self.eigd = np.linalg.eigh(inv_G_i @ G_io)
            self.__flip_eigd__()

class eK(TransfOp):
    def __init__(self, inp, outp, kern, regul : float = 0.00001, compute_eigd = False, logsp = False):
        TransfOp.__init__(self, inp, outp, kern, regul, compute_eigd, logsp)
        G_io = kern.gram(inp, outp)
        inv_G_io = np.linalg.inv(G_io + regul * np.eye(inp.shape[0]))
        G_o = kern.gram(outp)
        inv_G_i = np.linalg.inv(kern.gram(inp) + regul * np.eye(inp.shape[0]))
        self.matr = (inv_G_io @ inv_G_io.T @ G_o).T
        if compute_eigd:
            self.eigd = np.linalg.eigh(G_io.T @ inv_G_i)
            self.__flip_eigd__()
            
#class CovarianceOp:
#    def __init__(self, inp, kern, regul = 0.000001, cent = False):
#        self.inp = inp
#        self.kern = kern
#        self.G = kern(inp)
#        self.G_reg = self.G + regul * np.eye(self.G.shape[0])
#        if not cent:
#            self.eigVal_G, self.eigVec_G = np.linalg.eigh(self.G)
#        else:
#            c_mat = (np.eye(self.G.shape[0]) - np.ones(self.G.shape)/self.G.shape[0])/self.G.shape[0]
#            self.eigVal_G, self.eigVec_G = np.linalg.eigh(c_mat @ self.G)
#        self.eigVal_G, self.eigVec_G = np.flip(self.eigVal_G), np.fliplr(self.eigVec_G)
#        self.eval_datapoint_in_eigFunc = self.G @ self.eigVec_G
#        self.eval_datapoint_in_eigFunc_Sign = np.sign(self.eval_datapoint_in_eigFunc)
#        self.d_i_eF_argsort = np.array([np.argsort(eF) for eF in self.eval_datapoint_in_eigFunc.T]).T
#
#class CrossCov(TransfOp):
#    def __init__(self, inp, outp, kern, regul : float = 0.00001, compute_eigd = False, logsp = False):
#        TransfOp.__init__(self, inp, outp, kern, regul, compute_eigd, logsp)
#        self.matr = np.eye(inp.shape[0])/inp.shape[0]
#        if compute_eigd:
#            self.eigd = np.linalg.eigh(self.matr @ kern.gram(outp, inp))
#            self.__flip_eigd__()
#
#class Cov(TransfOp):
#    def __init__(self, inp, outp, kern, regul : float = 0.00001, compute_eigd = False, logsp = False):
#        outp = inp = np.vstack([inp, outp])
#        TransfOp.__init__(self, inp, outp, kern, regul, compute_eigd, logsp)
#        self.matr = np.eye(inp.shape[0])/inp.shape[0]
#        if compute_eigd:
#            self.eigd = np.linalg.eigh(self.matr @ kern.gram(outp, inp))
#            self.__flip_eigd__()