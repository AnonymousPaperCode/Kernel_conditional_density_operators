# -*- coding: utf-8 -*-


from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv
from distributions import categorical

__all__ = ["resid_res"]




def resid_res(pop, log_weights, weight_signs, resampled_size = None):
    if resampled_size is None:
        resampled_size = len(pop)        
    prop_w = np.array(log_weights) - logsumexp(log_weights)
    mult = exp(prop_w + log(resampled_size))
    count = np.uint32(np.floor(mult))
    resid = log(mult - count)
    resid = resid - logsumexp(resid)
    count = count + np.random.multinomial(resampled_size - count.sum(), exp(resid))
          
    return  (np.repeat(pop, count, axis = 0), np.repeat(weight_signs, count, axis = 0))
        
    
    