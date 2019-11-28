#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np, scipy as sp, scipy.stats as stats

from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from rkhsop.tools.linalg import logdotexp



def approximate_expectation(test_function, nsamples_per_support, support_points, factors, kernel, test_result_size = None, coupled_sampling = False, return_var = False, noise_factor = 1):
    factors = density_norm(factors.squeeze(),axis = 0)
    assert(len(factors.shape) <= 2)
    assert(factors.shape[0] == support_points.shape[0])
    
    samps = np.repeat(support_points[:, None, :], nsamples_per_support, 1)
    if coupled_sampling:
        assert('Coupled sampling is bogus')
        noise = kernel.rvs(nsamples_per_support, support_points.shape[1])[None, :, :]* noise_factor
    else:
        noise = (kernel.rvs(1, samps.size).reshape(*samps.shape))* noise_factor
    #print(noise.mean(), noise.std())
    samps = samps + (noise )
    
    evaluated_test_func = test_function(samps.reshape(nsamples_per_support * support_points.shape[0], -1))
    evaluated_test_func = evaluated_test_func.reshape(support_points.shape[0], nsamples_per_support, -1,  1)
    evaluated_test_func = np.swapaxes(evaluated_test_func, 0, 1)
    #return factors.reshape(1, support_points.shape[0], 1, -1)
    factors = factors.reshape(1, support_points.shape[0], 1, -1)/ nsamples_per_support
    
    if test_result_size is not None:
        assert(evaluated_test_func.shape[2] == test_result_size)

    mul = evaluated_test_func * factors
    moment_1 = np.sum(mul, axis = (0, 1))
    
    if return_var:    
        moment_2 = np.sum(evaluated_test_func*mul, axis = (0, 1))
        #assert()
        return (moment_1, (moment_2 - moment_1**2))
    #assert()
    return moment_1

def approximate_expectation_1d(test_function, nsamples_per_support, support_points, factors, kernel, coupled_sampling = True):
    if coupled_sampling:
        noise = kernel.rvs(nsamples_per_support, support_points.shape[1])
        samps = (np.tile(support_points,(noise.shape[0], 1)) + np.repeat(noise, support_points.shape[0], 0)).reshape(noise.shape[0], support_points.shape[0], support_points.shape[1])
    else:
        samps = np.tile(support_points,(nsamples_per_support, 1))
        samps = samps + kernel.rvs(*samps.shape)
    return  np.sum(test_function(np.vstack(samps)).reshape(nsamples_per_support, support_points.shape[0], -1) * factors.reshape(1, -1, 1) / nsamples_per_support,
                    axis = (0, 1))


def test_approximate_expectation():
    from rkhsop.kern.base import GaussianKernel
    support_points = np.array([(1,1), (2,2)])
    factors = np.array(([-1.799999, 1.8],[0.5, 0.5])).T
    k = GaussianKernel(0.3)
    print(support_points)
    print(factors)
    
    #approximate_expectation_1d(lambda x: x, 1000, support_points, factors[:, 0], k)
    np.allclose(approximate_expectation(lambda x: x, 2000, support_points, factors, k), np.array([[1.8, 1.5],[1.8  , 1.5]]), rtol=1, atol = 0.08)


if False:
    if False:
        def approximate_expectation(test_function, nsamples_per_support, support_points, factors, kernel):
            noise = kernel.rvs(nsamples_per_support, support_points.shape[1])
            samps = (np.tile(support_points,(noise.shape[0], 1)) + np.repeat(noise, support_points.shape[0], 0)).reshape(noise.shape[0], support_points.shape[0], support_points.shape[1])
            return  np.sum(test_function((samps)).reshape(noise.shape[0], support_points.shape[0], -1) * factors.reshape(1, -1, 1) / noise.shape[0],
                            axis = (0, 1))
    else:
        def approximate_expectation(test_function, nsamples_per_support, support_points, factors, kernel):
            factors = factors.squeeze()
            assert(len(factors.shape) == 1)
            assert(factors.shape[0] == support_points.shape[0])

            noise = kernel.rvs(nsamples_per_support, support_points.shape[1])
            samps = (np.tile(support_points,(noise.shape[0], 1)) + np.repeat(noise, support_points.shape[0], 0)).reshape(noise.shape[0], support_points.shape[0], support_points.shape[1])
            test_shape = (noise.shape[0], support_points.shape[0], -1)
            fact_shape = (1, -1, 1)

            return  np.sum(test_function((samps)).reshape(*test_shape) * factors.reshape(*fact_shape) / noise.shape[0],
                            axis = (0, 1))

def approximate_density(test_points, support_points, factors, kernel, logspace = False, component_selection = None):
    G = kernel.rkhsel_gram(support_points, np.atleast_2d(test_points), logsp = logspace).T
    if not logspace:
        factors = factors.copy()
        if component_selection is not None:
            # component_selection > 0 for only positive part
            # component_selection < 0 for only negative part
            idx = (np.sign(factors) == -np.sign(component_selection))
            factors[idx] = 0.
        return np.squeeze(G@factors)
    else:
        assert(component_selection is None) #doesnt work yet for logspace
        return logdotexp(G, factors.squeeze())

def density_norm(prefactors, signs = None, logsp = False, axis = None):
    if not logsp:
        s = prefactors.sum(axis, keepdims = True)
        assert(not np.any(np.isnan(s)) and not np.any(np.isinf(s)))
        return prefactors / s
    else:
        assert(axis is None) # normaxis doesnt work yet for logspace
        assert(not np.any(np.isnan(prefactors)) and not np.any(np.isinf(prefactors)))
        if signs is None:
            return prefactors - logsumexp(prefactors, keepdims = True)
        else:
            norm = logsumexp(prefactors, b = signs, keepdims = False, return_sign = True)
            return (prefactors - norm[0], signs * norm[1])

