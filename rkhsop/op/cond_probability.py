#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np, scipy as sp, scipy.stats as stats

from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from numpy import exp, log, sqrt
from scipy.misc import logsumexp

from rkhsop.op.base import RKHSOperator

class ConditionMeanEmbedding(RKHSOperator):
    def __init__(self, inp, outp, inp_kern, outp_kern, inp_regul = 0.00001):
        RKHSOperator.__init__(self, inp, outp, inp_kern, outp_kern)
        G_inp = inp_kern.gram(inp)
        self.matr = np.linalg.inv(G_inp+inp_regul * np.eye(G_inp.shape[0]))


class ConditionDensityOperator(RKHSOperator):
    def __init__(self, inp, outp, L_outp, inp_kern, outp_kern, inp_regul = 0.00001, outp_regul = 0.00001):
        RKHSOperator.__init__(self, inp, L_outp, inp_kern, outp_kern)
        G_l = outp_kern.gram(L_outp)
        
        G_inp_inv = np.linalg.inv(inp_kern.gram(inp) + inp.shape[0] * inp_regul * np.eye(inp.shape[0]))
        G_l_inv_factor = np.linalg.inv((G_l + outp_regul * np.eye(L_outp.shape[0])) @
                                       (G_l + outp_regul * np.eye(L_outp.shape[0])))
        G_l_outp = outp_kern.rkhsel_gram(outp, L_outp).T
        self.matr =  L_outp.shape[0]**2 * G_l_inv_factor @  G_l_outp @ G_inp_inv
        

def regul(nsamps, nrefsamps = None, a = 0.49999999999999, b = 0.49999999999999, c = 0.1):
    """smaller c => larger bias, tighter stochastic error bounds
       bigger  c =>  small bias, looser stochastic error bounds"""
    if nrefsamps is None:
        nrefsamps = nsamps
        
    assert(a > 0 and a < 0.5)
    assert(b > 0 and b < 0.5)
    assert(c > 0 and c < 1)
    assert(nsamps > 0 and nrefsamps > 0)
    
    return max(nrefsamps**(-b*c), nsamps**(-2*a*c))
