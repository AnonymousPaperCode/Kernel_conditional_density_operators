#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#from __future__ import division, print_function, absolute_import
from __future__ import print_function



import numpy as np

from scipy.optimize import minimize

from scipy.special import logsumexp
from numpy import exp, log

from rkhsop.kern.base import Kernel
from rkhsop.tools.linalg import logdotexp
from rkhsop.tools.rkhsdensity import approximate_density, density_norm





#def variational_loss(target, kernel):
#    def loss(at_points, log_prefactors):
#        targ_evals = target(at_points)
#        pref = exp(log_prefactors)
#        return np.sum(pref * targ_evals.flatten()) - np.sum(np.outer(pref, pref) * kernel.gram(at_points))
##        return logsumexp(log_prefactors + targ_evals.flatten()) - logsumexp(np.add.outer(log_prefactors, log_prefactors) + kernel.gram(at_points, logsp=True))
#    return loss

class RKHSOperator(object):
    def __init__(self, inp, outp, inp_kern : Kernel, outp_kern : Kernel, logsp = False):
        self.inp = inp
        self.logsp = logsp
        self.outp = outp
        self.inp_kern = inp_kern
        self.outp_kern = outp_kern

    def opmul(self, other, in_logsp = False):
        assert(self.inp_kern == other.outp_kern)
        assert(not in_logsp)
        # if in_logsp:
        #     loggram = self.inp_kern.gram(self.outp, other.inp, logsp = True)
        #     (tmp, tmp_sign) = logdotexp(log(np.abs(self.matr))), loggram, np.sign(self.matr), return_sign = True)
        #     (tmp, tmp_sign) = logdotexp(tmp, log(np.abs(other.matr)), tmp_sign, np.sign(other.matr), return_sign = True)
        rval = RKHSOperator(other.inp, self.outp, other.inp_kern, self.outp_kern)
        rval.matr = self.matr @ self.inp_kern.gram(self.inp, other.outp) @ other.matr
        return rval


    def lhood(self, inp, outp, diag = False, logsp = False, inp_mean_embedd = False, mean_embedd_fact = None, normalize = False):
        fact = self.get_factors(inp, inp_mean_embedd, mean_embedd_fact)
        if normalize:
            fact = density_norm(fact, axis=0)
        rval = approximate_density(outp, self.outp, density_norm(fact, logsp=self.logsp, axis=0), self.outp_kern, self.logsp)
        if diag and len(rval.shape) > 1:
            rval = np.diag(rval)
        if self.logsp == logsp:
            return rval
        elif logsp:
            return log(rval)
        else:
            return exp(rval)

#    def expected_value(self, function, at_inputs, num_samples_per_inducing):
#        f = density_norm(self.get_factors(at_inputs), axis = 0)
#        samps = (self.outp_kern.rvs(num_samples_per_inducing * self.outp.shape[0], self.outp.shape[1])
#                                + np.repeat(self.outp, num_samples_per_inducing, 0))
#        func_samps = function(samps).reshape(f.shape[0], num_samples_per_inducing) # needs to be samples x inputs x dimensionality
#        mult = func_samps[:,:, np.newaxis] * f[:, np.newaxis, :]    #?? mult = f.T @ func_samps 
        #print(f.shape, samps.shape, func_samps.shape, mult.shape)
#        return mult.sum((0,1))/num_samples_per_inducing
        
        
    def get_lhood_mean_var(self, inp, outp):
        assert(not self.logsp)
        samps = self.outp_kern.rkhsel_gram(self.outp, np.atleast_2d(outp)).T
        weights = np.tile(density_norm(self.get_factors(inp), logsp=False, axis=0), (samps.shape[0], 1, 1))
        mean = (samps[:,:,np.newaxis]*weights).sum(1)
        var = (samps[:,:,np.newaxis]**2*weights).sum(1) - mean**2
        print(weights.shape, samps.shape, weights.var(1).mean(), density_norm(self.get_factors(inp), logsp=False, axis=0).var(), var.mean(), (samps[:,:,np.newaxis]*weights).var(1).mean())
        return mean, var


    def get_factors(self, inp, inp_mean_embedd = False, mean_embedd_fact = None):
        if not self.logsp:
            gram = self.inp_kern.gram(self.inp, inp.reshape(-1, self.inp.shape[1]))
            if inp_mean_embedd:
                if mean_embedd_fact is None:
                    gram = gram.mean(1)
                else:
                    gram = np.sum(gram * np.repeat(mean_embedd_fact.flatten()[np.newaxis, :], gram.shape[0], 0), 1)
            fact = self.matr @ gram
        else:
            loggram = self.inp_kern.gram(self.inp, inp.reshape(-1, self.inp.shape[1]), logsp = self.logsp)
            if inp_mean_embedd:
                if mean_embedd_fact is None:
                    loggram = logsumexp(loggram, 1) - log(loggram.shape[1])
                else:
                    loggram = logsumexp(loggram + np.repeat(log(mean_embedd_fact).flatten()[np.newaxis, :], loggram.shape[0], 0), 1)
            fact = logdotexp(self.logmatr, loggram)
        return fact

    def mean_var(self, inp):
        fact = density_norm(self.get_factors(inp), logsp=self.logsp, axis=0).T
        #return(fact)
        #fact = self.get_factors(inp).T
        if self.logsp:
            fact = exp(fact)
        print(self.outp.shape, fact.shape, (fact @ self.outp).shape)
        mean = fact @ self.outp #np.sum(fact @ self.outp, 1)
        expected_variance = self.outp_kern.get_var()
        variance_expectations = (fact @ (self.outp**2)) - mean**2 #np.sum(fact @ self.outp**2, 1) - mean**2
        print(expected_variance, variance_expectations)
        return (mean, expected_variance + variance_expectations)
