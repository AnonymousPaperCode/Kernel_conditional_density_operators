#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np, scipy as sp, scipy.stats as stats

from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from rkhsop.test.banana import Banana

def test_rkhsoperator_matrparams(D=1, nsamps = 200, num_induc = 20, fit_matr = True, fit_kern = True, fit_induc = True, triang_matrix = False):
    targ = Banana(2, )
    samps = targ.rvs(nsamps)
    (out_samps, inp_samps) = (samps[:, :1], samps[:, 1:])

    o = MatrKerIndDensityOp(inp_samps, out_samps, ro.GaussianKernel(1), ro.GaussianKernel(1),  num_induc, fit_matr = fit_matr, fit_kern = fit_kern, fit_induc = fit_induc, triang_matrix = triang_matrix)
    gk_x = ro.GaussianKernel(1)
    gk_y = ro.GaussianKernel(1)
    
    x = np.linspace(-20,20,200)

    cdo = ro.ConditionDensityOperator(inp_samps, out_samps, gk_y, gk_x, 5, 5)


    print("CDO:", cdo.lhood(inp_samps, out_samps, diag = True).sum(), "\n", "IO :", (o.lhood(inp_samps, out_samps, diag = True)).sum())
    

    o.fit(inp_samps, out_samps)
    if num_induc == 4:
        o.inp = np.array([[0,1, 5, 5.01]]).T
        o.outp = np.array([[0, 0, -2, 2]]).T
    print("GF :", (o.lhood(inp_samps, out_samps, diag = True).sum()))
    d = cdo.lhood(np.array([[-1.], [5.]]), x[:, None]).T
    g = o.lhood(np.array([[-1.], [5.]]), x[:, None], logsp = True).T
    g = exp(g - logsumexp(g, 1)[:, np.newaxis] + 2.5) 
    t = np.vstack([targ.logpdf(np.hstack((x[:, None], -1 * np.ones((x.size, 1))))),
                   targ.logpdf(np.hstack((x[:, None],  5 * np.ones((x.size, 1)))))])
    t = exp(t - logsumexp(t, 1)[:, np.newaxis] + 2.5)
    
    
    (fig, ax) = pl.subplots(3, 1, True, False, figsize=(10,10))
    ax[2].scatter(out_samps, inp_samps, alpha=0.3)
    ax[2].axhline(-1, 0, 8, color='r', linestyle='--')
    ax[2].axhline(5, 0, 8, color='r', linestyle='--')
    ax[2].set_title("Input:  y, output: x,  %d pairs"%nsamps)
    ax[2].set_yticks((-1,5))
#    ax[1].plot(x, d[1], '-', label='cond. density')
    ax[1].plot(x, g[1,:], '--', label='cond. density by gradient descent')
    ax[1].plot(x, t[1,:], '--', label='Truth')
    ax[1].set_title("p(x|y=5)")
    print(cdo.mean_var(np.array([[5., -1]]).T), o.mean_var(np.array([[5., -1]]).T))
#    ax[0].plot(x, d[0], '-', label='cond. density')
    ax[0].plot(x, g[0, :], '--', label='cond. density by gradient descent')
    ax[0].plot(x, t[0,:], '--', label='Truth')
    ax[0].set_title("p(x|y=-1)")
    ax[0].legend(loc='best')
    fig.show()
    return (o,cdo, inp_samps, out_samps)  