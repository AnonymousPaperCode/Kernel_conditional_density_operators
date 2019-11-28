#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np, scipy as sp, scipy.stats as stats

from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from numpy import exp, log, sqrt
from scipy.misc import logsumexp

from rkhsop.op.base import RKHSOperator

from rkhsop.tools.linalg import logdotexp, assert_equal_log
from rkhsop.tools.rkhsdensity import approximate_density, density_norm

from rkhsop.tools.resampling import resid_res
from scipy.stats import multivariate_normal as mvn
import distributions as dist



class OptimizationPreimageOperator(RKHSOperator):
    def __init__(self, inp, kern):
        RKHSOperator.__init__(self, inp, inp, kern, kern, logsp = True)
        self.logmatr = np.random.randn(inp.shape[0], inp.shape[0])/30
        #self.__train()

    def train(self, unif, maxiter = 10, embedded_inp = False, bound = 4):
        assert(self.logsp)
        gram_out = gram_in = self.inp_kern(self.inp, unif, logsp=True)
        if embedded_inp:
            gram_in = logsumexp(gram_in, 1, keepdims = True)
        def llhood(logmatr):
            dotprod = logdotexp(logmatr.reshape(self.logmatr.shape), gram_in)
#            assert(not (np.any(np.isnan(dotprod)) or np.any(np.isinf(dotprod))))
            return -logsumexp(gram_out + dotprod, 0).sum()
        res = minimize(llhood, self.logmatr.flatten(), jac = grad(llhood), bounds=[(-bound, bound)]*self.logmatr.size, options = {'maxiter':maxiter}, callback = lambda x: print("\r", int(llhood(x)), end='   ', flush=True))
        self.logmatr = res['x'].reshape(self.logmatr.shape)
        return res

class PreimageOperator(RKHSOperator):
    def __init__(self, refsamples, kern, regul):
        RKHSOperator.__init__(self, refsamples, refsamples, kern, kern)
        self.matr = np.linalg.inv(kern.gram(refsamples) + np.eye(len(refsamples)) * regul)
        self.matr = self.matr @ self.matr

class IsPreimgOp(RKHSOperator): #using importance sampling
    def __init__(self, kern, regul, imp_dens = None, refsamples = None, data_smp = None, kde_kern = None, method = 'kde'):
        if method == 'kde':
            refsamples = data_smp + kde_kern.rvs(self.samps.shape[0]).reshape(*self.samps.shape)
            smp_pdf = exp(dist.location_mixture_logpdf(refsamples, data_smp, np.ones(data_smp.shape[0]), kde_kern))
        elif method == 'direct_is':
            smp_pdf = imp_dens(refsamples)
        RKHSOperator.__init__(self, refsamples, refsamples, kern, kern)
        self.matr = np.linalg.inv(kern.gram(refsamples) + np.eye(len(refsamples)) * regul)
        self.matr = (self.matr * smp_pdf[np.newaxis, :]) @ self.matr

class RmPreimgOp(RKHSOperator): #using reference measure
    def __init__(self, kern, regul, ref_dens = None, refsamples = None, data_smp = None, kde_kern = None, method = 'kde'):
        if method == 'kde':
            refsamples = data_smp + kde_kern.rvs(self.samps.shape[0]).reshape(*self.samps.shape)
            self.ref_dens = lambda smp: exp(dist.location_mixture_logpdf(smp, data_smp, np.ones(data_smp.shape[0]), kde_kern))
        elif method == 'direct_is':
            self.ref_dens = ref_dens
        self.matr = np.linalg.inv(kern.gram(refsamples) + np.eye(len(refsamples)) * regul)
        self.matr = self.matr @ self.matr

    def opmul(self, other : RKHSOperator, in_logsp = False):
            assert(self.inp_kern == other.outp_kern)
            assert(not in_logsp)
            rval = RKHSOperator(other.inp, self.outp, other.inp_kern, self.outp_kern)
            rval.matr = self.matr @ self.inp_kern.gram(self.inp, other.outp) @ other.matr * self.ref_dens(other.outp).reshape(1, -1)
            return rval

class RKHSDensityEstimator(object):
    def __init__(self, samps, kern, regul, kde_ref_meas_lenghtscale = None):
        self.regul = regul
        self.samps = samps
        self.kern = kern
        self.log_G = kern.gram(samps, logsp = True)
        self.G = exp(self.log_G)
        self.G_i = np.linalg.inv(self.G + np.eye(self.G.shape[0]) * regul)
        self.log_G_i = (np.abs(self.G_i), np.sign(self.G_i))
        self.kme_fact = density_norm(np.ones(samps.shape[0])/samps.shape[0])
        self.unif_dens_fact = self.G_i.sum(0)
        self.norm_const = self.unif_dens_fact.sum()
        self.unif_dens_fact = self.unif_dens_fact #/ self.norm_const
        print(self.unif_dens_fact.sum(), self.unif_dens_fact.shape, 1./self.unif_dens_fact.size)
#        self.dens_fact = density_norm((self.G_i @ np.diag(1./(self.G_i.mean(0))) @ self.G).mean(1))
        self.dens_fact = density_norm(1./self.G_i.mean(0))
        self.dens_inv_fact = (self.G_i @ self.G_i).mean(0)
        if kde_ref_meas_lenghtscale is not None:
            self.density_kern = mvn(np.zeros(self.samps.shape[1]), np.eye(self.samps.shape[1]) * kde_ref_meas_lenghtscale**2)
            self.pert_samps = self.samps + self.density_kern.rvs(self.samps.shape[0]).reshape(*self.samps.shape)
            self.unif_samps = resid_res(self.samps, log(np.abs(self.unif_dens_fact)), np.sign(self.unif_dens_fact))
#            assert()
            self.unif_samps = (self.unif_samps[0] + self.density_kern.rvs(self.samps.shape[0]).reshape(*self.samps.shape), self.unif_samps[1])
            self.pert_G_i = np.linalg.inv(self.kern.gram(self.pert_samps) + np.eye(self.pert_samps.shape[0]) * self.regul)
            self.log_pert_G_i = (log(np.abs(self.pert_G_i)), np.sign(self.pert_G_i))

            self.log_G_pert_samps = self.kern.gram(self.pert_samps, self.samps, logsp = True)
            self.G_pert_samps = exp(self.log_G_pert_samps)

            self.log_kde_pdf_at_pert = dist.location_mixture_logpdf(self.pert_samps, self.samps, np.ones(self.samps.shape[0]), self.density_kern)
            self.kde_pdf_at_pert = exp(self.log_kde_pdf_at_pert)

            self.log_kde_pdf_at_samps = dist.location_mixture_logpdf(self.samps, self.samps, np.ones(self.samps.shape[0]), self.density_kern)
            self.kde_pdf_at_samps = exp(self.log_kde_pdf_at_samps)


            unif_pert_G_i = np.linalg.inv(self.kern.gram(self.unif_samps[0]) + np.eye(self.unif_samps[0].shape[0]) * self.regul)
            unif_G_pert_samps = self.kern.gram(self.unif_samps[0], self.samps)
            self.density_fact_un = density_norm((unif_pert_G_i @ np.diag(self.unif_samps[1]) @ unif_pert_G_i @ unif_G_pert_samps).mean(1))

            self.density_fact_is = density_norm((self.pert_G_i @ np.diag(self.kde_pdf_at_pert) @ self.pert_G_i @ self.G_pert_samps).mean(1))

            log_df_is = logdotexp(self.log_pert_G_i[0] + self.log_kde_pdf_at_pert[np.newaxis, :], self.log_pert_G_i[0], sign_A = self.log_pert_G_i[1], sign_B = self.log_pert_G_i[1], return_sign = True)
            assert_equal_log(log_df_is, self.pert_G_i @ np.diag(self.kde_pdf_at_pert) @ self.pert_G_i)
            log_df_is = logdotexp(log_df_is[0], self.log_G_pert_samps, sign_A = log_df_is[1], return_sign = True)
            assert_equal_log(log_df_is, self.pert_G_i @ np.diag(self.kde_pdf_at_pert) @ self.pert_G_i @ self.G_pert_samps)

            self.log_density_fact_is = logsumexp(log_df_is[0], 1, b = log_df_is[1], return_sign = True)
            self.log_density_fact_is = density_norm(*self.log_density_fact_is, logsp = True)


            #assert_equal_log(self.log_density_fact_is, self.density_fact_is)

            self.density_fact_refmeas = density_norm(self.pert_G_i @ self.pert_G_i @ self.G_pert_samps @ self.kde_pdf_at_samps)


    def eval_kde(self, at_points):
        #evaluate kernel density estimator, which is the same as kernel mean embedding
        return self.eval_kme(at_points, logsp)

    def eval_kme(self, at_points):
        return approximate_density(at_points, self.samps, self.kme_fact, self.kern)

    def eval_rkhs_uniform_approx(self, at_points, component_selection = None):
        return approximate_density(at_points, self.samps, self.unif_dens_fact, self.kern, component_selection = component_selection)

    def eval_rkhs_uniform_embd_approx(self, at_points):
        return approximate_density(at_points, self.samps, self.unif_dens_fact, self.kern.get_double_var_kern())

#        K_mix = self.kern.gram(at_points, self.samps)
#        rval = (K_mix @ self.G_i)
#        print('own', rval.shape)
#        rval = rval.mean(1)
#        return rval *0.1/rval.max()

    def sample_rkhs_density_approx(self, nsamps = 1, is_estimator = True):
        assert(nsamps >= 1)
        idx = np.argmax(self.log_density_fact_is[0][:, np.newaxis] + np.random.gumbel(loc=0, scale=1, size = (self.log_density_fact_is[0].size, nsamps)), 0)
#        assert()
        return (self.pert_samps[idx] + self.kern.rvs(nsamps, self.pert_samps.shape[1]), self.log_density_fact_is[1][idx])

    def eval_rkhs_density_approx(self, at_points, uniform_samples = None, estimator = 'rm', component_selection = None):
        if uniform_samples is not None:
            samples = uniform_samples
            K_i = np.linalg.inv(self.kern.gram(uniform_samples) + np.eye(len(uniform_samples)) * self.regul)
            K_unif_rho = self.kern.gram(uniform_samples, self.samps)
            factors = (K_i @ K_i @ K_unif_rho).sum(1)/self.norm_const
        else:
            samples = self.pert_samps
            if estimator == 'rm':
                factors = self.density_fact_refmeas
            elif estimator == 'is':
                factors = self.density_fact_is
            elif estimator == 'un':
                factors = self.density_fact_un
            elif estimator == 'tr':
                samples = self.samps
                factors = 1./self.G_i.mean(0)
        factors = density_norm(factors)

        return approximate_density(at_points, samples, factors, self.kern, component_selection = component_selection)

    def eval_rkhs_density_inv_approx(self, at_points):
        return self.kern.gram(at_points) @ self.dens_inv_fact
