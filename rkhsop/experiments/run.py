#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%

import numpy as np, scipy as sp, scipy.stats as stats
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from numpy import exp, log, sqrt
from scipy.misc import logsumexp

from rkhsop.op.base import RKHSOperator
from rkhsop.op.cond_probability import ConditionDensityOperator, ConditionMeanEmbedding
from rkhsop.op.preimg_densest import IsPreimgOp, RmPreimgOp
import rkhsop.kern.base as kern
from rkhsop.experiments.data import Power, DataPrep, Traffic, Mountain, Jura
import matplotlib.pyplot as plt

import sklearn.gaussian_process.kernels as sklkern

def power(idx = 0, inp_bandwidth = 0.5, op = "cdo", plot = 'full', train_fraction = 0.05, only_cov = True):
    
    ts_idx = Power().get_ihgp_window(idx)['Global_active_power']
    (d, ts_idx) = (ts_idx.values, ts_idx.index.values)
    if only_cov:
        indepvar = ts_idx.reshape(-1, 1)
        depvar = d.reshape(-1, 1)
        
        inp_kern = kern.SKlKernel( sklkern.ExpSineSquared(length_scale = 1, periodicity=60) + # minute
                                   sklkern.ExpSineSquared(length_scale = 1, periodicity=60 * 60) + # hour
                                   sklkern.ExpSineSquared(length_scale = 1, periodicity=60 * 60 * 24) + # day
                                   sklkern.ExpSineSquared(length_scale = 1, periodicity=60 * 60 * 24 * 7) #+ # week
                                  #sklkern.ExpSineSquared(length_scale = 1, periodicity=60 * 24 * 30.41)  # month
                                  )
        #inp_kern = kern.SKlKernel(sklkern.RBF(1))
    else:
        (indepvar, depvar) = DataPrep.prepare_markov_1st(ts_idx.reshape(-1, 1), d.reshape(-1, 1))
        
        inp_kern = kern.SplitDimsKernel([0, 1, 2], [kern.SKlKernel(#sklkern.ExpSineSquared(length_scale = 1, periodicity=60) + # minute
                                                                   sklkern.ExpSineSquared(length_scale = 1, periodicity=60 * 60) + # hour
                                                                   sklkern.ExpSineSquared(length_scale = 1, periodicity=60 * 60 * 24) + # day
                                                                   sklkern.ExpSineSquared(length_scale = 1, periodicity=60 * 60 * 24 * 7) #+ # week
                                                                   #sklkern.ExpSineSquared(length_scale = 1, periodicity=60 * 24 * 30.41)  # month
                                                                  ),
                                               kern.SKlKernel(sklkern.RBF(1))])
        # inp_kern = kern.SKlKernel(sklkern.RBF(1))
      
    train_end = int(train_fraction * len(indepvar))
    (train_inp, test_inp) = (indepvar[:train_end, :], indepvar[train_end:, :])
    (train_out, test_out) = (depvar[:train_end, :], depvar[train_end:, :])
    for v in [indepvar, train_inp, test_inp, depvar, train_out, test_out]:
        print(v.shape, end = '')
    print()

    L_samp = np.linspace(d.min()-0.2, d.max()+0.2, len(train_out)).reshape(-1, 1)
    
    
    out_kern = kern.GaussianKernel(np.abs(L_samp[0] - L_samp[1]) * 50)
    if op == 'cdo':
        cdo = ConditionDensityOperator(train_inp, train_out, L_samp,
                                                inp_kern, out_kern,
                                                inp_regul=0.001, outp_regul=0.0001)
    elif op == 'cmo':
        cdo = ConditionMeanEmbedding(train_inp, train_out,
                                     inp_kern, out_kern,
                                     inp_regul=0.001)
    else:
        assert('operator "'+str(op)+'" unknown')
    print("Used", train_end, "samples for training", len(d), "for testing")
    
    x = np.vstack([train_inp, test_inp])

    if plot == 'full':
        dens_eval = cdo.lhood(x, L_samp)
        rnge = max(np.abs(dens_eval.min()), np.abs(dens_eval.max()))
        plt.imshow(dens_eval, origin = 'lower', cmap='RdBu',
                   extent = (ts_idx[0], ts_idx[-1], L_samp[0, 0], L_samp[-1, 0]), vmin = -rnge, vmax = rnge, aspect='auto', alpha = 0.5)
        plt.colorbar()
    else:
        (m, v) = cdo.mean_var(x)
        sd = np.sqrt(v)
        plt.plot(x.T[0], m, "b--", alpha = 0.5)
        plt.fill_between(ts_idx, m + 2 * sd, m - 2 * sd, color='r', alpha=0.2)
    plt.scatter(ts_idx, d, color = 'k', alpha = 0.2)

def traffic(idx = 0, inp_bandwidth = 0.5, op = "cdo", plot = 'full', samps_per_timepoint = 5, only_cov = True):
    dimslice = slice(2, 3)
    traf = Traffic()
    traf.data['train'] = Traffic.equalize_smp_size(traf.data['train'])
    traf.data['test'] = traf.data['test']

    (train_inp, train_out) = Traffic.stackup_smp(traf.data['train'])
    train_out = train_out[:, slice(dimslice.start- 2, dimslice.stop- 2, dimslice.step) , :samps_per_timepoint]
    # assert()
    if only_cov:
        (test_inp, test_out) = (traf.data['test'][['DoW', 'Sec of Day']].values, traf.data['test'].values[:, dimslice])
        
        inp_kern = kern.SplitDimsKernel([0, 1, 2], [kern.SKlKernel(sklkern.ExpSineSquared(length_scale = 1, periodicity=7)), # weekly period
                                                    kern.SKlKernel(sklkern.ExpSineSquared(length_scale = 1, periodicity=24 * 60 * 60))]) # daily period
        inp_kern = kern.SplitDimsKernel([0, 1, 2], [kern.SKlKernel(sklkern.RBF(1)), kern.SKlKernel(sklkern.RBF(60*60))])
    else:
        (train_inp, train_out) = DataPrep.prepare_markov_1st(train_inp, train_out)
        (test_inp, test_out) = DataPrep.prepare_markov_1st(traf.data['test'][['DoW', 'Sec of Day']].values, traf.data['test'].values[:, dimslice])
        inp_kern = kern.SplitDimsKernel([0, 1, 2], [kern.SKlKernel(#sklkern.ExpSineSquared(length_scale = 1, periodicity=60) + # minute
                                                                   sklkern.ExpSineSquared(length_scale = 1, periodicity=60 * 60) + # hour
                                                                   sklkern.ExpSineSquared(length_scale = 1, periodicity=60 * 60 * 24) + # day
                                                                   sklkern.ExpSineSquared(length_scale = 1, periodicity=60 * 60 * 24 * 7) #+ # week
                                                                   #sklkern.ExpSineSquared(length_scale = 1, periodicity=60 * 24 * 30.41)  # month
                                                                  ),
                                               kern.SKlKernel(sklkern.RBF(1))])
        # inp_kern = kern.SKlKernel(sklkern.RBF(1))

    for v in [ train_inp, test_inp, train_out, test_out]:
        print(v.shape, end = '')
    print()

    L_samp = np.linspace(train_out.min()-0.2, train_out.max()+0.2, len(train_out)).reshape(-1, 1)
    
    
    out_kern = kern.GaussianKernel(np.abs(L_samp[0] - L_samp[1]) * 5)
    if op == 'cdo':
        cdo = ConditionDensityOperator(train_inp, train_out, L_samp,
                                                inp_kern, out_kern,
                                                inp_regul=0.001, outp_regul=0.0001)
    elif op == 'cmo':
        cdo = ConditionMeanEmbedding(train_inp, train_out,
                                     inp_kern, out_kern,
                                     inp_regul=0.001)
    else:
        assert('operator "'+str(op)+'" unknown')
    print("Used", len(train_out), "samples for training", len(test_out), "for testing")
    
    
    # x = np.hstack([np.repeat(np.arange(7), 144)[:, None], np.tile(np.arange(144), 7)[:, None]])
    x = np.hstack([np.zeros(144)[:, None], np.arange(144)[:, None] * self.measurement_interval_in_sec])
    lim = [0, 144]

    if plot == 'full':
        dens_eval = cdo.lhood(x, L_samp)
        rnge = max(np.abs(dens_eval.min()), np.abs(dens_eval.max()))
        plt.imshow(dens_eval, origin = 'lower', cmap='RdBu',
                    extent = (lim[0], lim[1], L_samp[0, 0], L_samp[-1, 0]),
                   vmin = -rnge, vmax = rnge, aspect='auto', alpha = 0.5)
        plt.colorbar()
        # assert()
    else:
        (m, v) = cdo.mean_var(x)
        sd = np.sqrt(v)
        plt.plot(x.T[0], m, "b--", alpha = 0.5)
        plt.fill_between(ts_idx, m + 2 * sd, m - 2 * sd, color='r', alpha=0.2)
    plt.scatter(ts_idx, d, color = 'k', alpha = 0.2)

def traffic_diff(idx = 0, inp_bandwidth = 0.5, op = "cdo", plot = 'full', mean_plot=False, samps_per_timepoint = 5):
    dimslice = slice(2, 3)
    traf = Traffic()
    traf.data['train'] = Traffic.diff(Traffic.equalize_smp_size(traf.data['train']))
    traf.data['test'] = Traffic.diff(traf.data['test'])

    (train_inp, train_out) = Traffic.stackup_smp(traf.data['train'])
    train_out = train_out[:, slice(dimslice.start- 2, dimslice.stop- 2, dimslice.step) , :samps_per_timepoint]
    (test_inp, test_out) = (traf.data['test'][['DoW', 'Sec of Day']].values, traf.data['test'].values[:, dimslice])
    
    inp_kern = kern.SplitDimsKernel([0, 1, 2], [kern.SKlKernel(sklkern.ExpSineSquared(length_scale = 1, periodicity=7)), # weekly period
                                                kern.SKlKernel(sklkern.ExpSineSquared(length_scale = 1, periodicity=24 * 60 * 60))]) # daily period
    inp_kern = kern.SplitDimsKernel([0, 1, 2], [kern.SKlKernel(sklkern.ExpSineSquared(length_scale = 1, periodicity=7)), # weekly period
                                                kern.SKlKernel(sklkern.RBF(60*60))])


    for v in [ train_inp, test_inp, train_out, test_out]:
        print(v.shape, end = '')
    print()

    L_samp = np.linspace(train_out.min()*1.02, train_out.max()*1.02, len(train_out)).reshape(-1, 1)
    print('Interval', (train_out.min(), train_out.max()))
    
    
    out_kern = kern.GaussianKernel(np.abs(L_samp[0] - L_samp[1]) * 5)
    # assert()
    if op == 'cdo':
        cdo = ConditionDensityOperator(train_inp, train_out, L_samp,
                                                inp_kern, out_kern,
                                                inp_regul=0.001, outp_regul=0.0001)
    elif op == 'cmo':
        cdo = ConditionMeanEmbedding(train_inp, train_out,
                                     inp_kern, out_kern,
                                     inp_regul=0.001)
    else:
        assert('operator "'+str(op)+'" unknown')
    print("Used", len(train_out), "samples for training", len(test_out), "for testing")
    
    
    x = np.hstack([np.repeat(np.arange(7), 143)[:, None], np.tile(np.arange(1, 144)*600, 7)[:, None]])
    # x = np.hstack([np.zeros(144)[:, None], np.arange(144)[:, None]])
    lim = [0, 143*7]

    sample_traj = Traffic.equalize_smp_size(traf.data['test'],143).sort_values(["DoW", "Sec of Day"]).values[:, dimslice].flatten()

    test_statistics = np.zeros(x.shape)
    i = 0
    for g in Traffic.equalize_smp_size(traf.data['test']).sort_values(["DoW", "Sec of Day"]).groupby(["DoW", "Sec of Day"]):
        test_statistics[i,:] = (g[1].values[:, dimslice].mean(), g[1].values[:, dimslice].std())
        
        i = i + 1
    # assert()
    
    dens_eval = cdo.lhood(x, L_samp)
    rnge = max(np.abs(dens_eval.min()), np.abs(dens_eval.max()))
    plt.imshow(dens_eval, origin = 'lower', cmap='RdBu',
                extent = (lim[0], lim[1], L_samp[0, 0], L_samp[-1, 0]),
                vmin = -rnge, vmax = rnge, aspect='auto', alpha = 0.5)
    plt.colorbar()
    plt.scatter(np.arange(lim[1]), test_statistics.T[0], alpha = 0.3)
    
    
    if mean_plot:
        plt.plot(np.arange(lim[1]), test_statistics.T[0], "r-", alpha = 0.3)
        plt.plot(np.arange(lim[1]), test_statistics.T[0] + test_statistics.T[1], "r--", alpha = 0.3,)
        plt.plot(np.arange(lim[1]), test_statistics.T[0] - test_statistics.T[1], "r--", alpha = 0.3,)
        plt.xticks(np.tile(np.arange(0, 143, 20), 7) + np.repeat(np.arange(7) * 143, 8), list(range(0, 143, 20)) * 7)
    else:
        
        sort_traf = Traffic.equalize_smp_size(traf.data['test']).sort_values(["DoW", "Sec of Day"]).values[:, dimslice].flatten()
        num_rep = len(sort_traf) // 143
        plt.scatter(np.tile(np.arange(143), num_rep) + np.repeat(np.arange(7) * 143, len(sort_traf)//7), sort_traf, alpha = 0.3)
    for i in range(7):
        plt.axvline(i * 143)

def mountain(train_fraction = 0.9, inp_bandwidth = 0.5, op = "cdo", inp_regul = 0.1, outp_regul = 0.1, cv_iters = 10, compute_gp = True, compute_op = True, compute_flow = True, learning_rate = 1e-8, epochs= 100):
    

    m = Mountain().data
    trainSize = int(np.ceil(train_fraction*np.prod(m['z'].shape)))
    testSize = int(np.prod(m['z'].shape)) - trainSize
    orig_perm = perm = np.load('permutation.npy').flatten() #np.random.permutation(np.prod(m['z'].shape))
    coord = np.hstack((m['x'].reshape(-1, 1), m['y'].reshape(-1, 1)))
    elev = m['z'].reshape(-1, 1)
    grad = m['fin_diff'].reshape(-1, 2)
    if compute_op:
        smae_op = np.empty(cv_iters)
    if compute_gp:
        smae_gp = np.empty(cv_iters)
    if compute_flow:
        smae_flow = np.empty(cv_iters)
    res = {'permutation':perm, "testSize": testSize}
    for i in range(1):
        print("Iteration", i)
        Itest = perm[:testSize]
        Itrain = perm[testSize:]
        perm = np.roll(perm, testSize)
        if False:
            print( Itest.size, testSize, Itrain.size, trainSize, perm.size)
            #assert(i==0)
        else:
            res = {'permutation':orig_perm, "testSize": testSize}

            train_inp, test_inp = coord[Itrain,:], coord[Itest,:]
            train_out, test_out = elev[Itrain,:], elev[Itest,:]
            print("Train/Testsets")
            train_mean = np.mean(train_out)
            train_out, test_out = train_out - train_mean, test_out - train_mean
            print("standardization")
            if compute_op:
                inp_kern = kern.GaussianKernel(np.sqrt(np.median(pdist(train_inp)))/2) #kern.SKlKernel(sklkern.RBF(10, (1e-2, 1e2)))
                print("lengthscale")
                L_samp = np.linspace(train_out.min()-0.2, train_out.max()+0.2, len(train_out)).reshape(-1, 1)
                out_kern = kern.GaussianKernel(np.abs(L_samp[0] - L_samp[1]) * 50)
                print("Output kernel")
                if op == 'cdo':
                    cpo = ConditionDensityOperator(train_inp, train_out, L_samp,
                                                            inp_kern, out_kern,
                                                            inp_regul=inp_regul, outp_regul=outp_regul)
                elif op == 'cmo':
                    cpo = ConditionMeanEmbedding(train_inp, train_out,
                                                inp_kern, out_kern,
                                                inp_regul=inp_regul)
                print("trained")
                (pred_mean, pred_var) = cpo.mean_var(test_inp)
                smae_op[i] = np.abs(pred_mean - test_out.T).sum()/np.abs(test_out).sum()
                print("Op: %.4f (±%.4f), last: %.4f" % (smae_op[:i+1].mean(), smae_op[:i+1].std(), smae_op[i]))
                res["smaeOp"] = smae_op[:i+1]
            if compute_gp:
                #it might be beneficial for the GP to Z-transform the data
                if True:
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    kernel_GP = sklkern.RBF(2*(np.array([3.56728274, 5.35779433])**2))
                    gp = GaussianProcessRegressor(kernel=kernel_GP, alpha = inp_regul)

                    # Fit to data using Maximum Likelihood Estimation of the parameters
                    gp.fit(train_inp, train_out)
                    gp_pred = gp.predict(test_inp)
                else:
                    import exact_gpytorch as egp
                    gp = egp.ExactGPModel.optimized_gp(train_inp, train_out.flatten()) 
                    gp_pred = gp.pred_mean(test_inp)

                smae_gp[i] = np.abs(gp_pred - test_out).sum()/np.abs(gp_pred).sum()
                print("GP: %.4f (±%.4f), last: %.4f" % (smae_gp[:i+1].mean(), smae_gp[:i+1].std(), smae_gp[i]))
                res["smaeGP"] = smae_gp[:i+1]
            if compute_flow:
                from rkhsop.experiments.conditional_flow import CondFlow
                condflow = CondFlow.optimized_cde(train_inp.astype(np.float), train_out.astype(np.float), learning_rate = learning_rate, batch_size=train_inp.shape[0]//10, epochs = epochs, weight_decay=1)
                
                mean_pred = []
                for j in test_inp:
                    mean_pred.append(condflow.sample(10, j.astype(np.float)).mean().cpu().detach().numpy())
                mean_pred = np.array(mean_pred)
                smae_flow[i] = np.abs(mean_pred - test_out).sum()/np.abs(mean_pred).sum()
                print("Flow: %.4f (±%.4f), last: %.4f" % (smae_flow[:i+1].mean(), smae_flow[:i+1].std(), smae_flow[i]))
                res["smaeFlow"] = smae_flow[:i+1]
            
            sp.io.savemat('cde_experiment_flow.mat', res)
    
def mountain_old(train_fraction = 0.9, inp_bandwidth = 0.5, op = "cdo", inp_regul = 0.1, outp_regul = 0.1, cv_iters = 10):
    from sklearn.gaussian_process import GaussianProcessRegressor
    m = Mountain().data
    trainSize = int(np.ceil(train_fraction*np.prod(m['z'].shape)))
    testSize = int(np.prod(m['z'].shape)) - trainSize
    perm = np.random.permutation(np.prod(m['z'].shape))
    coord = np.hstack((m['x'].reshape(-1, 1), m['y'].reshape(-1, 1)))
    elev = m['z'].reshape(-1, 1)
    grad = m['fin_diff'].reshape(-1, 2)
    smae_op = np.empty(cv_iters)
    # = np.empty(cv_iters)
    for i in range(cv_iters):
        Itest = perm[i * testSize:(i + 1) * testSize]
        Itrain = np.hstack([perm[0:i * testSize], perm[(i + 1) * testSize:]])
        print(Itest.size, Itrain.size, Itest.size + Itrain.size == perm.size)
        print(i)
        if True:
            train_inp, test_inp = coord[Itrain,:], coord[Itest,:]
            train_out, test_out = elev[Itrain,:], elev[Itest,:]
            train_mean = np.mean(train_out)
            train_out, test_out = train_out, test_out

            inp_kern = kern.SKlKernel(sklkern.RBF(11, (1e-2, 1e2))) # median/2 is about 11 everytime
            L_samp = np.linspace(train_out.min()-0.2, train_out.max()+0.2, len(train_out)).reshape(-1, 1)
            out_kern = kern.GaussianKernel(np.abs(L_samp[0] - L_samp[1]) * 50)
            if op == 'cdo':
                cpo = ConditionDensityOperator(train_inp, train_out, L_samp,
                                                        inp_kern, out_kern,
                                                        inp_regul=inp_regul, outp_regul=outp_regul)
            elif op == 'cmo':
                cpo = ConditionMeanEmbedding(train_inp, train_out,
                                            inp_kern, out_kern,
                                            inp_regul=inp_regul)
            (pred_mean, pred_var) = cpo.mean_var(test_inp)
            
            smae_op[i] = np.abs(pred_mean - test_out.T).sum()/np.abs(test_out).sum()
            print("Op", smae_op[i])

            kernel_GP = sklkern.RBF(2*(np.array([3.56728274, 5.35779433])**2))
            gp = GaussianProcessRegressor(kernel=kernel_GP, alpha = inp_regul)

            # Fit to data using Maximum Likelihood Estimation of the parameters
            gp.fit(train_inp, train_out)
            gp_pred = gp.predict(test_inp)

            smae_gp[i] = np.abs(gp_pred - test_out).sum()/np.abs(gp_pred).sum()
            print("GP", smae_gp[i])
    print("GP:", smae_gp.mean(), smae_gp.std())
    print("Op:", smae_op.mean(), smae_op.std())
    sp.io.savemat('cde_experiment_old.mat', {'permutation':perm, "testSize": testSize, "smaeGP": smae_gp, "smaeOp": smae_op})

def jura(train_fraction = 0.9, inp_bandwidth = 0.5, op = "cdo", inp_regul = 0.1, outp_regul = 0.1, cv_iters = 10):
    from sklearn.gaussian_process import GaussianProcessRegressor
    jur = Jura().data
    trainSize = int(np.ceil(train_fraction*np.prod(m['z'].shape)))
    testSize = int(np.prod(m['z'].shape)) - trainSize
    perm = np.random.permutation(np.prod(m['z'].shape))
    coord = np.hstack((m['x'].reshape(-1, 1), m['y'].reshape(-1, 1)))
    elev = m['z'].reshape(-1, 1)
    grad = m['fin_diff'].reshape(-1, 2)
    smae_op = np.empty(cv_iters)
    smae_gp = np.empty(cv_iters)
    for i in range(cv_iters):
        Itest = perm[i * testSize:(i + 1) * testSize]
        Itrain = np.hstack([perm[0:i * testSize], perm[(i + 1) * testSize:]])
        #print(Itest, Itrain, Itest.size + Itrain.size, perm.size)

        if True:
            train_inp, test_inp = coord[Itrain,:], coord[Itest,:]
            train_out, test_out = elev[Itrain,:], elev[Itest,:]
            train_mean = np.mean(train_out)
            train_out, test_out = train_out - train_mean, test_out - train_mean

            inp_kern = kern.GaussianKernel(np.sqrt(np.median(pdist(train_inp)))/2)
            L_samp = np.linspace(train_out.min()-0.2, train_out.max()+0.2, len(train_out)).reshape(-1, 1)
            out_kern = kern.GaussianKernel(np.abs(L_samp[0] - L_samp[1]) * 50)
            if op == 'cdo':
                cpo = ConditionDensityOperator(train_inp, train_out, L_samp,
                                                        inp_kern, out_kern,
                                                        inp_regul=inp_regul, outp_regul=outp_regul)
            elif op == 'cmo':
                cpo = ConditionMeanEmbedding(train_inp, train_out,
                                            inp_kern, out_kern,
                                            inp_regul=inp_regul)
            (pred_mean, pred_var) = cpo.mean_var(test_inp)
            smae_op[i] = np.abs(pred_mean - test_out.T).sum()/np.abs(test_out).sum()

            kernel_GP = sklkern.RBF(2*(np.array([3.56728274, 5.35779433])**2))
            gp = GaussianProcessRegressor(kernel=kernel_GP, alpha = inp_regul)

            # Fit to data using Maximum Likelihood Estimation of the parameters
            gp.fit(train_inp, train_out)
            gp_pred = gp.predict(test_inp)

            smae_gp[i] = np.abs(gp_pred - test_out).sum()/np.abs(gp_pred).sum()
    print("GP:", smae_gp.mean(), smae_gp.std())
    print("Op:", smae_op.mean(), smae_op.std())
    sp.io.savemat('cde_experiment.mat', {'permutation':perm, "testSize": testSize, "smaeGP": smae_gp, "smaeOp": smae_op})
    
def old_mountain(train_fraction = 0.9, inp_bandwidth = 0.5, op = "cdo", inp_regul = 0.1, outp_regul = 0.1, cv_iters = 10):
    from sklearn.gaussian_process import GaussianProcessRegressor
    m = Mountain().data
    trainSize = int(np.ceil(train_fraction*np.prod(m['z'].shape)))
    testSize = int(np.prod(m['z'].shape)) - trainSize
    perm = np.random.permutation(np.prod(m['z'].shape))
    coord = np.hstack((m['x'].reshape(-1, 1), m['y'].reshape(-1, 1)))
    elev = m['z'].reshape(-1, 1)
    grad = m['fin_diff'].reshape(-1, 2)
    smae_op = np.empty(cv_iters)+np.nan
    smae_gp = np.empty(cv_iters)+np.nan
    for i in range(cv_iters):
        Itest = perm[i * testSize:(i + 1) * testSize]
        Itrain = np.hstack([perm[0:i * testSize], perm[(i + 1) * testSize:]])
        print(Itest.size, Itest[[0,-1]], Itrain.size, Itrain[[0, -1]], Itest.size + Itrain.size, perm.size)
        print(np.unique(Itest).size + np.unique(Itrain.size)-perm.size)
        train_inp, test_inp = coord[Itrain,:], coord[Itest,:]
        train_out, test_out = elev[Itrain,:], elev[Itest,:]
        train_mean = np.mean(train_out)
        train_out, test_out = train_out - train_mean, test_out - train_mean
        median_lengthscale = np.sqrt(np.median(pdist(train_inp)))/2

        if False:
            print(median_lengthscale)
        else:
            inp_kern = kern.SKlKernel(sklkern.RBF(11, (1e-2, 1e2))) # median heuristic is almost 11 every time kern.GaussianKernel(median_lengthscale)
            L_samp = np.linspace(train_out.min()-0.2, train_out.max()+0.2, len(train_out)).reshape(-1, 1)
            out_kern = kern.GaussianKernel(np.abs(L_samp[0] - L_samp[1]) * 50)
            if op == 'cdo':
                cpo = ConditionDensityOperator(train_inp, train_out, L_samp,
                                                        inp_kern, out_kern,
                                                        inp_regul=inp_regul, outp_regul=outp_regul)
            elif op == 'cmo':
                cpo = ConditionMeanEmbedding(train_inp, train_out,
                                            inp_kern, out_kern,
                                            inp_regul=inp_regul)
            (pred_mean, pred_var) = cpo.mean_var(test_inp)
            smae_op[i] = np.abs(pred_mean - test_out.T).sum()/np.abs(test_out).sum()


            
            print("Op:", smae_op[:i+1].mean(), smae_op[:i+1].std())
            sp.io.savemat('old_mountain_experiment.mat', smae_op[:i+1])
