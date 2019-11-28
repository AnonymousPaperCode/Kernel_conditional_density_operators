#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np, scipy as sp, scipy.stats as stats

from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from numpy import exp, log, sqrt
from scipy.misc import logsumexp

def logdotexp(A, B, sign_A = None, sign_B = None, return_sign=False):
    assert(A.shape[-1] == B.shape[0])
    s = A+B.T.reshape(-1, 1, B.shape[0])
    if sign_A is None and sign_B is None:
        rval = logsumexp(s, -1, return_sign = return_sign)
    else:
        if sign_B is None:
            rval = logsumexp(s, -1, b = sign_A.reshape(1, *A.shape), return_sign = return_sign)
        else:
            sign_B = sign_B.T.reshape(-1, 1, B.shape[0])
            if sign_A is None:
                rval = logsumexp(s, -1, b = sign_B, return_sign = return_sign)
            else:
                rval = logsumexp(s, -1, b = sign_A * sign_B, return_sign = return_sign)
    #        if sign_A is None:
    #            sign_A = np.ones_like(A)
    #        if sign_B is None:
    #            sign_B = np.ones_like(B)
    #        sg = sign_A * sign_B.T.reshape(-1, 1, B.shape[0])
    #        rval = logsumexp(s, -1, b = sg, return_sign = return_sign)
    if return_sign:
        return (rval[0].T, rval[1].T)
    else:
        return rval.T

def assert_equal_log(logsp, standardsp):
    assert(np.max(np.abs( exp(logsp[0]) * logsp[1] - standardsp)) < 1e-5)

def test_logdotexp():
    for _ in range(10):
        A = np.random.randn(4,2)
        B = np.random.randn(2,3)
        for sign_A in (True, False):
            for sign_B in (True, False):
                kwargs = {"return_sign":True}
                if not sign_A:
                    mA = np.abs(A)
                else:
                    mA = A
                    kwargs["sign_A"] = np.sign(A)
                if not sign_B:
                    mB = np.abs(B)
                else:
                    mB = B
                    kwargs["sign_B"] = np.sign(B)

                ls = logdotexp(log(np.abs(A)),log(np.abs(B)), **kwargs)
                standard_r = mA @ mB
                assert_equal_log(ls, standard_r)
