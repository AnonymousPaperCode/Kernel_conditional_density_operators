#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np, scipy as sp, scipy.stats as stats

from scipy.optimize import minimize

from autograd.numpy import exp, log, sqrt
from autograd.scipy.misc import logsumexp
from scipy.special import erf

import autograd.numpy as np
import autograd.scipy as sp
from autograd import grad

import autograd.scipy.stats as stats


from rkhsop.op.preimg_densest import RKHSDensityEstimator
from rkhsop.op.cond_probability import ConditionMeanEmbedding, ConditionDensityOperator #, Corrected2ConditionDensityOperator
from rkhsop.kern.base import GaussianKernel, LaplaceKernel, StudentKernel

import pylab as pl

import distributions as dist
from rkhsop.tools.resampling import resid_res

def test_variational(n_points):
    D = 1
    targ = dist.mixt(D, [dist.mvnorm(3*np.ones(D), np.eye(D)*0.7**2), dist.mvnorm(7*np.ones(D), np.eye(D)*1.5**2)], [0.5, 0.5])
    targ_pdf = lambda x: exp(targ.logpdf(x))

    l = variational_loss(targ_pdf, GaussianKernel(1))
    x_init = (np.random.randn(n_points) * 5).reshape((n_points, 1))
    log_pref_init = np.random.randn(n_points) * 5

    g_x = grad(l, 0)
    g_p = grad(l, 1)
    lrate = 0.001
    for i in range(10):
        print(l(x_init, log_pref_init))
        dx = g_x(x_init, log_pref_init)
        dp = g_p(x_init, log_pref_init)
        x_init = x_init + lrate * dx
        log_pref_init = log_pref_init + lrate * dp
    print(x_init, exp(log_pref_init))

def test_rkhs_unif(D = 1, nsamps = 200, samps_unif = None, regul_dens = 0.001):
    from matplotlib import rc
    if samps_unif is None:
        samps_unif = nsamps
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

#    targ = MoU()
    targ = dist.mixt(D, [dist.mvnorm(3*np.ones(D), np.eye(D)*0.7**2), dist.mvnorm(7*np.ones(D), np.eye(D)*1.5**2)], [0.5, 0.5])
    out_samps = targ.rvs(nsamps).reshape([nsamps, 1])

    gk_x = GaussianKernel(0.7)
    x = np.linspace(-2.5, 15, samps_unif)

    de = RKHSDensityEstimator(out_samps, gk_x, regul_dens, kde_ref_meas_lenghtscale = 0.9)

    pl.figure(figsize = (7.5, 3))
    pl.plot(x, exp(targ.logpdf(x)), 'k--', alpha = 0.5, label=r'\textrm{True} $\rho$')
    pl.plot(x, de.eval_rkhs_uniform_approx(x[:,None]), 'b-', label=r'$\widehat{p}_{U}$')
    pos_dens = de.eval_rkhs_uniform_approx(x[:,None], component_selection = +1)
    neg_dens = de.eval_rkhs_uniform_approx(x[:,None], component_selection = -1)

    scaling = 1 #0.3 / np.max(np.abs(np.stack((pos_dens, neg_dens))))
    pl.plot(x, pos_dens * scaling, "-", label=r'$\widehat{p}_{U}^+$')
    pl.plot(x, -neg_dens * scaling, "-",  label=r'$\widehat{p}_{U}^-$')
    assert(nsamps == samps_unif)
    pl.title("Estimates using %d samples" % (nsamps))
    pl.legend(loc='best')
    pl.tick_params(which='both', bottom=False, top=False, labelbottom=False)

    pl.tight_layout()
    dens_plot_fname = 'Uniform_estimation_%d_targSamps' % (nsamps)
    pl.savefig(dens_plot_fname + '.pdf')
    pl.savefig(dens_plot_fname + '.jpg')
    pl.close()

def rkhs_dens_plot_paper(D = 1, nsamps = 200, samps_unif = None, regul_dens = 0.001):
    from matplotlib import rc, rcParams

    rcParams.update({'font.size': 13})

    

    np.random.seed(0)
    if samps_unif is None:
        samps_unif = nsamps
    if regul_dens is None:
        regul_dens = np.max([(samps_unif ** -0.49), (nsamps ** -0.99)])
        print("Max from two sample numbers", regul_dens)
        regul_dens = 1.1 * regul_dens
        print("Regularizer", regul_dens)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

#    targ = MoU()
    targ = dist.mixt(D, [dist.mvnorm(3*np.ones(D), np.eye(D)*0.7**2), dist.mvnorm(7*np.ones(D), np.eye(D)*1.5**2)], [0.5, 0.5])
    out_samps = targ.rvs(nsamps).reshape([nsamps, 1])

    gk_x = GaussianKernel(0.9)
    #gk_x = LaplaceKernel(3)
    #gk_x = StudentKernel(0.7, 15)
    x = np.linspace(-2.5, 15, samps_unif)

    de = RKHSDensityEstimator(out_samps, gk_x, regul_dens, kde_ref_meas_lenghtscale = 1)
    pl.figure(figsize = np.array((7.5, 2.5)))
#        pl.scatter(samps_p, noise[:samps_p.size] / 4 + .3)
#        pl.scatter(samps_n, noise[:samps_n.size] / 4 - .3)
    pl.plot(x, exp(targ.logpdf(x)), 'k-', alpha = 1, label=r'$\textrm{True}~\frac{\mathrm{d}{P}}{\mathrm{d} \rho}$')
    #pl.plot(x, de.eval_rkhs_uniform_approx(x[:,None]), 'b--', label=r'$\widehat{p}_{U}$')
    #pl.scatter(out_samps, de.eval_rkhs_uniform_approx(out_samps))
    print("Min %.2f Max %.2f, Min uniform approx %.2f" % (out_samps.min(), out_samps.max(), de.eval_rkhs_uniform_approx(out_samps).min()))
    pl.plot(x, de.eval_rkhs_density_approx(x[:,None], x[:,None]), 'g:', label=r'$\widehat{\frac{\mathrm{d}{P}}{\mathrm{d} \rho}}$') #uniform reference
    #pl.plot(x, de.eval_rkhs_density_approx(x[:,None],  estimator = 'rm'), 'b--', alpha = 0.5, label=r'$\widehat{\rho}~(rm)$')
    #pl.plot(x, de.eval_rkhs_density_approx(x[:,None],  estimator = 'is'), 'r-.', alpha = 0.5, label=r'$\widehat{\rho}~(is)$')
    # pl.plot(x, de.eval_rkhs_density_approx(x[:,None],  estimator = 'tr'), 'r-.', alpha = 0.5, label=r'$\widehat{\rho}~(tr)$')
    pos_dens = de.eval_rkhs_density_approx(x[:,None],  estimator = 'un', component_selection = +1)
    neg_dens = de.eval_rkhs_density_approx(x[:,None],  estimator = 'un', component_selection = -1)

    scaling = 1 #0.3 / np.max(np.abs(np.stack((pos_dens, neg_dens))))
#        pl.plot(x, neg_dens + pos_dens,  alpha = 0.5, label=r'$\widehat{\rho}~(us)$')
#        pl.plot(x, pos_dens * scaling, "--", alpha = 0.5, label=r'$\widehat{\rho}^+~(is)$')
#        pl.plot(x, -neg_dens * scaling, "-.",  alpha = 0.5, label=r'$\widehat{\rho}^-~(is)$')
    pl.plot(x, de.eval_kme(x[:,None]), 'r-.', label=r'$\widehat{\mu}_{P}$')

    unif = de.eval_rkhs_uniform_approx(x[:,None])
    pl.plot(x, unif/unif.max() * 0.1, 'b-.', label=r'\textrm{support}')
    assert(nsamps == samps_unif)
    #pl.title("Estimates using %d samples" % (nsamps))
    pl.legend(loc='best')
    # pl.tick_params(which='both', bottom=False, top=False, labelbottom=False)

    pl.tight_layout()
    dens_plot_fname = 'Density_estimation_(preimage_of_KDE)_%d_targSamps_%d_unifSamps' % (nsamps, samps_unif)
    
    pl.savefig(dens_plot_fname + '.jpg')
    pl.savefig(dens_plot_fname + '.pdf')
    pl.close()

def test_rkhs_dens_and_operators(D = 1, nsamps = 200, samps_unif = None, test_dens = True, test_op = True, regul_dens = 0.001):
    from matplotlib import rc
    if samps_unif is None:
        samps_unif = nsamps
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

#    targ = MoU()
    targ = dist.mixt(D, [dist.mvnorm(3*np.ones(D), np.eye(D)*0.7**2), dist.mvnorm(7*np.ones(D), np.eye(D)*1.5**2)], [0.5, 0.5])
    out_samps = targ.rvs(nsamps).reshape([nsamps, 1])

    gk_x = GaussianKernel(0.9)
    gk_x = LaplaceKernel(3)
    #gk_x = StudentKernel(0.7, 15)
    x = np.linspace(-2.5, 15, samps_unif)

    if test_dens:
        de = RKHSDensityEstimator(out_samps, gk_x, regul_dens, kde_ref_meas_lenghtscale = 1)
    #
#        samps = de.sample_rkhs_density_approx(200)
#        samps_p = samps[0][samps[1] > 0].flatten()
#        samps_n = samps[0][samps[1] < 0].flatten()
#        noise = np.random.rand(samps[0].size)
    #    x = np.linspace(-6, 6, 3000)
        pl.figure(figsize = (7.5, 2.5))
#        pl.scatter(samps_p, noise[:samps_p.size] / 4 + .3)
#        pl.scatter(samps_n, noise[:samps_n.size] / 4 - .3)
        pl.plot(x, exp(targ.logpdf(x)), 'k-', alpha = 0.5, label=r'$\textrm{True}~\frac{\mathrm{d}{P}}{\mathrm{d} \rho}$')
        #pl.plot(x, de.eval_rkhs_uniform_approx(x[:,None]), 'b--', label=r'$\widehat{p}_{U}$')
        #pl.scatter(out_samps, de.eval_rkhs_uniform_approx(out_samps))
        print("Min %.2f Max %.2f, Min uniform approx %.2f" % (out_samps.min(), out_samps.max(), de.eval_rkhs_uniform_approx(out_samps).min()))
        pl.plot(x, de.eval_rkhs_density_approx(x[:,None], x[:,None]), 'g:', label=r'$\textrm{estim}~\frac{\mathrm{d}{P}}{\mathrm{d} \rho}$') #uniform reference
        #pl.plot(x, de.eval_rkhs_density_approx(x[:,None],  estimator = 'rm'), 'b--', alpha = 0.5, label=r'$\widehat{\rho}~(rm)$')
        #pl.plot(x, de.eval_rkhs_density_approx(x[:,None],  estimator = 'is'), 'r-.', alpha = 0.5, label=r'$\widehat{\rho}~(is)$')
        # pl.plot(x, de.eval_rkhs_density_approx(x[:,None],  estimator = 'tr'), 'r-.', alpha = 0.5, label=r'$\widehat{\rho}~(tr)$')
        pos_dens = de.eval_rkhs_density_approx(x[:,None],  estimator = 'un', component_selection = +1)
        neg_dens = de.eval_rkhs_density_approx(x[:,None],  estimator = 'un', component_selection = -1)

        scaling = 1 #0.3 / np.max(np.abs(np.stack((pos_dens, neg_dens))))
#        pl.plot(x, neg_dens + pos_dens,  alpha = 0.5, label=r'$\widehat{\rho}~(us)$')
#        pl.plot(x, pos_dens * scaling, "--", alpha = 0.5, label=r'$\widehat{\rho}^+~(is)$')
#        pl.plot(x, -neg_dens * scaling, "-.",  alpha = 0.5, label=r'$\widehat{\rho}^-~(is)$')
        pl.plot(x, de.eval_kme(x[:,None]), 'r-.', label=r'$\textrm{estim}~\mu_{P}$')

        unif = de.eval_rkhs_uniform_approx(x[:,None])
        pl.plot(x, unif/unif.max() * 0.1, 'b-.', label=r'$\textrm{estim}~0.1\frac{\mathrm{d}{P}}{\mathrm{d} {P}}$')
        assert(nsamps == samps_unif)
        pl.title("Estimates using %d samples" % (nsamps))
        pl.legend(loc='best')
       # pl.tick_params(which='both', bottom=False, top=False, labelbottom=False)

        pl.tight_layout()
        dens_plot_fname = 'Density_estimation_(preimage_of_KDE)_%d_targSamps_%d_unifSamps' % (nsamps, samps_unif)
        
        pl.savefig(dens_plot_fname + '.jpg')
        pl.savefig(dens_plot_fname + '.pdf')
        pl.close()
    #
    #    return

    if test_op:
        inp_samps = (out_samps-5)**2 + np.random.randn(*out_samps.shape)
        gk_y = GaussianKernel(1)

        cme = ConditionMeanEmbedding(inp_samps, out_samps, gk_y, gk_x, 5)
    #    cdo = ConditionDensityOperator(inp_samps, out_samps, gk_y, gk_x, 5, 5)
        cdo_u = ConditionDensityOperator(inp_samps, out_samps, np.linspace(out_samps.min()-1, out_samps.max()+1, out_samps.size)[:, None], gk_y, gk_x, 5, 5)
        cdo_t = Corrected2ConditionDensityOperator(inp_samps, out_samps, gk_y, gk_x, 5, 5)




        (fig, ax) = pl.subplots(3, 1, True, False, figsize=(10,10))
        ax[2].scatter(out_samps, inp_samps, alpha=0.3)
        ax[2].axhline(0, 0, 8, color='r', linestyle='--')
        ax[2].axhline(5, 0, 8, color='r', linestyle='--')
        ax[2].set_title("Input:  y, output: x,  %d pairs"%nsamps)
        ax[2].set_yticks((0, 5))
        d_u = cdo_u.lhood(np.array([[0.], [5.]]), x[:, None], diag = False, logsp = False).T
        d_t = cdo_t.lhood(np.array([[0.], [5.]]), x[:, None], diag = False, logsp = False).T
        e = cme.lhood(np.array([[0.], [5.]]), x[:, None], diag = False, logsp = False).T
        assert(d_u.shape[0] == 2)
    #    assert(np.allclose(d[0], cdo.lhood(0, x[:, None], diag = False, logsp = False)))
    #    assert(np.allclose(d[1], cdo.lhood(5, x[:, None], diag = False, logsp = False)))
    #    assert()
        ax[1].plot(x, d_u[1], '-', label='cond. density (using uniform samples)')
        ax[1].plot(x, d_t[1], '-', label='cond. density (target samples only)')
        ax[1].plot(x, e[1], '--', label='cond. mean emb.')
        ax[1].set_title(r"$p(x \mid y=5)$")
        ax[0].plot(x, d_u[0], '-', label='cond. density (using uniform samples)')
        ax[0].plot(x, d_t[0], '-', label='cond. density (target samples only)')
        ax[0].plot(x, e[0], '--', label='cond. mean emb.')
        ax[0].set_title(r"$p(x \mid y=0)$")
        ax[0].legend(loc='best')

        fig.savefig("conditional_density_operator.pdf")
#        fig.close()


def test_clustering(D = 1, nsamps = 200, samps_unif = None):
    from matplotlib import rc
    from rkhsop.tools.rkhsdensity import approximate_density

    if samps_unif is None:
        samps_unif = nsamps
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

#    targ = MoU()
    targ = dist.mixt(D, [dist.mvnorm(3*np.ones(D), np.eye(D)*0.7**2), dist.mvnorm(7*np.ones(D), np.eye(D)*1.**2)], [0.5, 0.5])
    out_samps = targ.rvs(nsamps).reshape([nsamps, 1])

    gk_x = GaussianKernel(0.7)
    G = gk_x.gram(out_samps)
    (evals, evecs) = np.linalg.eigh(G + 0.1 * np.eye(G.shape[0]))
    x = np.linspace(-2.5, 15, samps_unif)
#    x = np.linspace(-6, 6, 3000)

    pl.figure(figsize = (5, 2))
    pl.plot(x, exp(targ.logpdf(x)), 'k-', alpha = 0.5, label=r'\textrm{True} $\rho$')
    sc = exp(targ.logpdf(x)).max()/np.abs(approximate_density(x[:, np.newaxis], out_samps, np.real(evals[-1])*(evecs.T[-1]), gk_x)).max()
    for i in range(1,4):
        pl.plot(x, approximate_density(x[:, np.newaxis], out_samps, np.real(evals[-i])*sc*(evecs.T[-i]), gk_x), label=r'eigenfunction %d'%i)

    pl.title("%d target samples, %d uniform samples" % (nsamps, samps_unif))
    pl.legend(loc='best')
    pl.tick_params(which='both', bottom=False, top=False, labelbottom=False)
    pl.tight_layout()
    dens_plot_fname = 'Eigenfunctions'
    pl.savefig(dens_plot_fname + '.pdf')
    return (evals, evecs)


def test_rkhsoperator_logsp():
    from rkhsop.op.base import RKHSOperator

    a = np.arange(5).astype(np.float).reshape((5, 1))
    i = np.arange(3)
    o = RKHSOperator(a, a**2, inp_kern=GaussianKernel(1), outp_kern=GaussianKernel(1))
    o.matr = np.arange(1., 26.).reshape((5, 5))
    o.logmatr = log(o.matr)
    rs = np.random.RandomState(None)

    b = np.reshape(4 + rs.randn(3), (3,1))
    assert(np.allclose(o.lhood(2, b, False),
                          exp(o.lhood(2, b, True))))
    assert(np.allclose(o.lhood(np.arange(2).reshape((2, 1)) * 4 - 2, b, False),
                          exp(o.lhood(np.arange(2)  .reshape((2, 1)) * 4 - 2, b, True))))
