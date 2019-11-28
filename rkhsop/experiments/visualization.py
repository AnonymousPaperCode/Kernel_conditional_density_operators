#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:41:58 2019

"""

import numpy as np, scipy as sp, scipy.stats as stats

import collections
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import distributions as dist

from numpy import exp, log, sqrt
from scipy.misc import logsumexp

from rkhsop.op.cond_probability import CorrectedConditionDensityOperator, regul
from rkhsop.experiments.conditional_flow import CondFlow 

from rkhsop.kern.base import GaussianKernel, LaplaceKernel, median_heuristic

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from time import time
import pandas as pd
import cde.density_estimator as de


from rkhsop.experiments.conditional_flow import CondFlow
#
def pol2cart(theta, rho):
    x = (rho * np.cos(theta)).reshape(-1,1)
    y = (rho * np.sin(theta)).reshape(-1,1)
    return np.concatenate([x, y], axis = 1)

def cart2pol(x, y):
    theta = np.arctan2(y, x).reshape(-1,1)
    rho = np.hypot(x, y).reshape(-1,1)
    return np.concatenate([theta, rho], axis = 1)

def get_plane_coords(point, normal, rnge):

    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -point.dot(normal)

    # create x,y
    xx, yy = np.meshgrid(range(10), range(10))

    # calculate corresponding z
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    return xx, yy, z

def apply_to_mg(func, *mg):
    #apply a function to points on a meshgrid
    x = np.vstack([e.flat for e in mg]).T
    #assert()
    return np.array([func(i) for i in x]).reshape(mg[0].shape)

def cont(f,  coord, first_coord = None, grid_density=100):
    xx = np.linspace(coord[0][0], coord[0][1], grid_density)
    yy = np.linspace(coord[1][0], coord[1][1], grid_density)
    mg = list(np.meshgrid(xx,yy))
    if first_coord is None:
        c1, c2 = mg
    else:
        mg.insert(0, np.ones_like(mg[0]) * first_coord)
        c1, c2 = mg[1:]
    Z = apply_to_mg(f, *mg)
    #assert()
    fig,ax  = plt.subplots()
    cs = ax.contour(c1, c2, (Z))
    #ax.clabel(cs, inline=1, fontsize = 10)
    fig.show()
    return Z
    #plt.xlim(xmin=coord[0][0], xmax=coord[0][1])
    #plt.ylim(ymin=coord[1][0], ymax=coord[1][1])
   # fig.show()

def cont_mult(mg, heights,  grid_density=100, titles = None, vminmax = False):
    from matplotlib import rcParams
    rcParams.update({'font.size': 15})

    c1, c2 = mg
    nrows = 1
    ncols = heights.shape[0]

    cont_kwargs = {}
    if vminmax:
        cont_kwargs['vmin'] = heights.min()
        cont_kwargs['vmax'] = heights.max()
    
    #assert()
    fig,ax  = plt.subplots(nrows, ncols, sharex = True, sharey=True, figsize = (ncols*3.8, nrows*3))
    ax[0].set_xticks([-1,0,1])
    for c in range(heights.shape[0]):
        ax[c].contour(c1, c2, heights[c], **cont_kwargs)
        if titles is not None:
            ax[c].set_title(titles[c])
    #ax.clabel(cs, inline=1, fontsize = 10)
    fig.tight_layout()
    fig.show()
    return fig
    #plt.xlim(xmin=coord[0][0], xmax=coord[0][1])
    #plt.ylim(ymin=coord[1][0], ymax=coord[1][1])
   # fig.show()

def scatter_3d(samples):
    from matplotlib import rcParams
    rcParams.update({'font.size': 17})

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel(r"$x$")
    ax.set_xticks([-1, 0, 1])
    ax.set_ylabel("")
    ax.set_yticks([-1, 0, 1])
    ax.set_zlabel("")
    ax.set_zticks([-1, 0, 1])

    sze=100; XX = np.repeat(np.linspace(-1.5, 1.5, sze)[:,None], sze, 0)
    ax.plot_surface(np.ones_like(XX), XX, XX.T, rcount = 5, ccount = 5, color=[0.9]*3, alpha = 0.5)
    ax.plot_surface(np.zeros_like(XX), XX, XX.T, rcount = 5, ccount = 5, color = [0.9]*3, alpha = 0.5 )
    
    ax.scatter(*samples, depthshade=True)
    ax.view_init(azim=-84, elev = 20)
    #fig.tight_layout()
    fig.show()
    return fig

def compute_dens(funcs,  coord,  grid_density=100):
    xx = np.linspace(coord[0][0], coord[0][1], grid_density)
    yy = np.linspace(coord[1][0], coord[1][1], grid_density)
    mg = list(np.meshgrid(xx,yy))
    c1, c2 = mg
    if not isinstance(funcs, collections.Sequence):
        heights = np.array([[apply_to_mg(funcs, *mg)]])
    else:
        if not isinstance(funcs[0], collections.Sequence):
            heights = np.array([[apply_to_mg(f, *mg) for f in funcs]])
            ncols = len(funcs)
        else:
            nrows = len(funcs)
            ncols = len(funcs[0])
            heights = np.array([[apply_to_mg(f, *mg) for f in row] for row in funcs])
    return (mg, heights)

def plot_3d_and_cross_sections(mg, s1, s2, samples, sections, cmap = None, vminmax = False):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import rc, rcParams

    rcParams.update({'font.size': 13})

    fig = plt.figure(figsize = np.array((5*5, 3))/2)
    s1_t_ax = fig.add_subplot(1, 5, 1)
    s1_e_ax = fig.add_subplot(1, 5, 2, sharex = s1_t_ax, sharey = s1_t_ax)
    s2_t_ax = fig.add_subplot(1, 5, 4)
    s2_e_ax = fig.add_subplot(1, 5, 5, sharex = s2_t_ax, sharey = s2_t_ax)
    cont_kwargs = {}
    if cmap is not None:
        cont_kwargs['cmap'] = cmap
    if vminmax:
        cont_kwargs['vmin'] = np.min([s1.min(), s2.min()])
        cont_kwargs['vmax'] = np.max([s1.max(), s2.max()])
    for (ax, height, idx, truth) in [(s1_t_ax, s1[0], 0, True), (s1_e_ax, s1[1], 0, False), (s2_t_ax, s2[0], 1, True), (s2_e_ax, s2[1], 1, False)]:
        ax.contour(mg[0], mg[1], height, **cont_kwargs)
        if truth:
            title = r"truth"
        else:
            title = r"estimate"
        ax.set_title(title+r" at $x=%d$" % sections[idx])

    rcParams.update({'font.size': 10})
    ax = fig.add_subplot(1, 5, 3, projection='3d')
    ax.set_xlabel(r"$x$")
    ax.set_xticks([-1, 0, 1])
    ax.set_ylabel("")
    ax.set_yticks([-1, 0, 1])
    #ax.set_zlabel("")
    #ax.set_zticks([-1, 0, 1])

    sze=100; XX = np.repeat(np.linspace(-1.5, 1.5, sze)[:,None], sze, 0)
    ax.plot_surface(np.ones_like(XX), XX, XX.T, rcount = 5, ccount = 5, color=[0.9]*3, alpha = 0.5)
    ax.plot_surface(np.zeros_like(XX), XX, XX.T, rcount = 5, ccount = 5, color = [0.9]*3, alpha = 0.5 )
    
    ax.scatter(*samples, depthshade=True)
    ax.view_init(azim=-84, elev = 20)
    #fig.tight_layout()
    fig.show()
    return fig

def generate_donut(nmeans = 50, nsamps_per_mean = 50):
    rotation_angle = 0.5
    comp_distribution = multivariate_normal(np.zeros(3), np.eye(3)/100)
    means = np.concatenate([pol2cart(np.linspace(0,2*3.141, nmeans + 1)[:-1], 1), np.zeros((nmeans, 1))], axis = 1)
    means[:,1:] = pol2cart(*(cart2pol(*means.T[1:])+np.array([(rotation_angle, 0)])).T)
    #dist.mixture.location_mixture_logpdf(samps, means, np.ones(nmeans) / nmeans)
    rvs = comp_distribution.rvs(nmeans * nsamps_per_mean) + np.repeat(means, nsamps_per_mean, 0)
    true_dens = lambda samps: exp(dist.mixture.location_mixture_logpdf(samps, means, np.ones(nmeans) / nmeans, comp_distribution))
    return rvs, means, true_dens


def distances_donut(s = None, plot = False, recompute_regul = True, num_reps = 10,  num_epochs = 100, learning_rate_maf=0.005, learning_rate_rnvp=0.002):
    from sklearn.gaussian_process import GaussianProcessRegressor
    import sklearn.gaussian_process.kernels as sklkern
    #from KCEF.estimators.kcef import KCEF_Gaussian
                    
    from rkhsop.tools.array_shapes import dataset_for_kronecker_trick
    if s is None:
        s = int(time())
    print(s)
    np.random.seed(s)

    x1 = 1
    x2 = 0

    distance_refsamps = dataset_for_kronecker_trick(*([np.linspace(-2, 2, 50)[:, None]]*2))

    df = pd.DataFrame()
    nmeans = 50

    nsamps_values = np.array([11]) +5 
    nsamps_values = np.array([#10, 25, 50, 
                              75])

    c = 0.9999999999999 #loose stochastic bounds, but small approximation bias


    gp_d1 = GaussianProcessRegressor(kernel=sklkern.RBF([1.]), alpha = 0.1)
    gp_d2 = GaussianProcessRegressor(kernel=sklkern.RBF([1.]), alpha = 0.1)
    

    graph = [[[0], []],
             [[1, 2], [0]]]

    #kcef_est = KCEF_Gaussian(graph_type = 'custom', d = 3, graph = graph)
    #condflow  = CondFlow()
    
    
    est_order = ['CDO', 'GP', "CondRNVP", "CondMAF", "LSCDE"]

    for nsamps_per_mean in nsamps_values:
        nsamps = nmeans * nsamps_per_mean
        regul_dens = regul(int(nsamps_values.max() * nsamps_per_mean), c = c)
        for rep in range(num_reps):
            distance_refsamps = np.random.rand(3000,2)*4-2
            nsamps_per_refdim = 50
            sqrt_nsamps_ref = int(np.ceil(nsamps**0.5))
            print("nsamps", nsamps, "refsamps", sqrt_nsamps_ref**2)
            
            (rvs, means, true_dens) = generate_donut(nmeans, nsamps_per_mean)
            if recompute_regul:
                regul_dens = regul(int(nsamps), sqrt_nsamps_ref**2, c = c)
                print("Regularizer", regul_dens)

            #ref_samps = np.random.rand(nmeans*nsamps_per_mean,2)*4-2.
            ref_samps = dataset_for_kronecker_trick(*([np.linspace(-2, 2, sqrt_nsamps_ref)[:, None]]*2))

            # for (n, samps) in [("dat", rvs), ("ref", ref_samps)]:
            #     print(n, samps.mean(), samps.std(), samps.min(), samps.max())

            mmin = means.min(0)
            mmax = means.max(0)

            
            inp_lscale = out_lscale = 0.1
            inp_lscale = median_heuristic(rvs[:,:1], "euclidean", False)
            out_lscale = 3*(ref_samps.T[1,1] - ref_samps.T[1,0])
            
            inp_lscale = median_heuristic(rvs[:,:1], "euclidean", False)
            out_lscale = median_heuristic(ref_samps, "euclidean", False) 
            
            in_kern = LaplaceKernel(inp_lscale/np.sqrt(2))
            #in_kern = LaplaceKernel(inp_lscale/np.sqrt(2))
            out_kern = GaussianKernel(out_lscale)
            #out_kern = GaussianKernel(out_lscale)

            
            cdo = CorrectedConditionDensityOperator(rvs[:, :1], rvs[:, 1:], ref_samps, in_kern, out_kern, inp_regul=regul_dens, outp_regul=regul_dens)

            #kcef_est.fit(rvs)

            epochs = num_epochs
            batch_size = rvs.shape[0]#//10
            print("Rnvp")
            condrnvp    = CondFlow.optimized_cde(rvs[:, :1], rvs[:, 1:], epochs=epochs, batch_size=batch_size, learning_rate = learning_rate_rnvp/28*np.sqrt(len(rvs)))
            print('MAF')
            condmaf    = CondFlow.optimized_cde(rvs[:, :1], rvs[:, 1:], epochs=epochs, batch_size=batch_size, flow_type="maf", learning_rate = learning_rate_maf/28*np.sqrt(len(rvs)))
            gp_d1.fit(rvs[:, :1], rvs[:, 1])
            gp_d2.fit(rvs[:, :1], rvs[:, 2])

            perm = rvs[np.random.permutation(len(rvs))[:400], :].T

            gp_d1x1 = stats.norm(*gp_d1.predict(np.ones((1,1))*x1))
            gp_d1x2 = stats.norm(*gp_d1.predict(np.ones((1,1))*x2))
            gp_d2x1 = stats.norm(*gp_d2.predict(np.ones((1,1))*x1))
            gp_d2x2 = stats.norm(*gp_d2.predict(np.ones((1,1))*x2))
            
            lscde = de.LSConditionalDensityEstimation()
            lscde.fit_by_cv(rvs[:, :1], rvs[:, 1:]) # lscde.fit(rvs[:, :1], rvs[:, 1:], **{'bandwidth': 1.0, 'n_centers': 100, 'regularization': 1.0})

            #log_nc_x1 = kcef_est.log_pdf(distance_refsamps, np.ones((1,1))*x1, 1)
            #log_nc_x1 = logsumexp(log_nc_x1) - log(distance_refsamps.shape[0])
            #log_nc_x2 = kcef_est.log_pdf(distance_refsamps, np.ones((1,1))*x2, 1)
            #log_nc_x2 = logsumexp(log_nc_x2) - log(distance_refsamps.shape[0])

            
            (true_x1, est_x1, true_x2, est_x2) = [lambda samps: true_dens(np.hstack([x1, samps])),
                                                    lambda samps: np.array([cdo.lhood(np.ones((1,1))*x1, samps, normalize=True),
                                                                            gp_d1x1.pdf(samps[:,0])*gp_d2x1.pdf(samps[:,1]),

                                                                            exp(condrnvp.log_pdf(np.ones((len(samps),1))*x1, samps).cpu().detach().numpy().squeeze()),
                                                                            exp(condmaf.log_pdf(np.ones((len(samps),1))*x1, samps).cpu().detach().numpy().squeeze()),
                                                                            lscde.pdf(np.ones((len(samps),1))*x1, samps),
                                                                           ]),
                                                    lambda samps: true_dens(np.hstack([x2, samps])),
                                                    lambda samps: np.array([cdo.lhood(np.ones((1,1))*x2, samps, normalize=True),
                                                                            gp_d1x2.pdf(samps[:,0])*gp_d2x2.pdf(samps[:,1]),
                                                                            exp(condrnvp.log_pdf(np.ones((len(samps),1))*x2, samps).cpu().detach().numpy().squeeze()),
                                                                            exp(condmaf.log_pdf(np.ones((len(samps),1))*x2, samps).cpu().detach().numpy().squeeze()),
                                                                            lscde.pdf(np.ones((len(samps),1))*x2, samps),
                                                                           ])]
            for (name, tr, es) in [(x1, true_x1, est_x1), (x2, true_x2, est_x2)]:
                
                target_vals = np.array([tr(s) for s in distance_refsamps])
                est_vals = es(distance_refsamps).T
                dists = np.abs(est_vals- target_vals)
                rel_dists = np.abs(est_vals - target_vals) / np.abs(target_vals)
                iw = (target_vals / est_vals)
                for (dist_type, dist_val) in [('L2', np.sqrt(np.mean(dists**2, 0))),
                                              ('L1', np.mean(dists, 0)),
                                              ('sL1', np.mean(rel_dists, 0)),
                                              ('ISstd', np.sqrt(np.abs(np.mean(iw**2 * est_vals, 0) - np.mean(target_vals, 0)**2))),
                                              ('KL_targ_approx', np.mean(np.log(np.abs(iw))*target_vals, 0)),
                                              ('KL_approx_targ', np.mean(-np.log(np.abs(iw))*est_vals, 0))]:

                    for i, est_name in enumerate(est_order):
                        df = df.append({"repetition":rep,
                                        "num_samps":int(nsamps),
                                        "Distance":dist_type,
                                        "dist_val":dist_val[i],
                                        "est_name":est_name,
                                        "Location":name}, ignore_index=True)
    # for x in [x1, x2]:        
    #     for dist in ["L1", "L2"]:          
    #         for ns in res:            
    #             print(ns, dist, x, np.mean(res[ns][dist][x]), np.std(res[ns][dist][x]))

   # assert()
    if plot:
        boxplot_donut_distances(df, "L1")
    return df

def boxplot_donut_distances(df, dist_type = 'L1', dist_name = "L1 distance", showfliers=True):
    import seaborn as sns
    for loc in df.Location.unique():
        sns.set_palette(sns.color_palette("Paired"))
        fig = plt.figure(figsize=(10, 4))
        ax = sns.boxplot(data= df[np.bitwise_and(df["Distance"] == dist_type, df["Location"] == loc)], x = 'num_samps', y='dist_val', hue='est_name', showfliers=showfliers)
        if loc != 1.:
            ax.get_legend().remove()
        else:
            ax.get_legend().set_title("Estimator")
            ax.legend(loc="best", fontsize="x-small")
        ax.set_title('Location: x = %d'%loc)
        ax.set_xlabel('Estimates using %d samples' % int(df.num_samps.max()))
        ax.set_ylabel(dist_name)
        ax.set_xticks([])
        fig.tight_layout()
        fig.savefig('Donut_at_%d.pdf'%loc)
        plt.show()


def donut(s = 1558197780, plot = True):
    from rkhsop.tools.array_shapes import dataset_for_kronecker_trick
    if s is None:
        s = int(time())
    print(s)
    np.random.seed(s)
    nmeans = 50
    nsamps_per_mean = 100
    nsamps = nmeans * nsamps_per_mean
    nsamps_per_refdim = int(np.sqrt(nsamps))

    
    (rvs, means, true_dens) = generate_donut(nmeans, nsamps_per_mean)

    #ref_samps = np.random.rand(nmeans*nsamps_per_mean,2)*4-2.
    ref_samps = dataset_for_kronecker_trick(*([np.linspace(-2, 2, nsamps_per_refdim)[:, None]]*2))

    for (n, samps) in [("dat", rvs), ("ref", ref_samps)]:
        print(n, samps.mean(), samps.std(), samps.min(), samps.max())

    mmin = means.min(0)
    mmax = means.max(0)

    a = 0.5 - 10e-10
    b = 0.5 - 10e-10
    c = 1 - 10e-10
    regul_dens = np.min([nsamps, nsamps_per_refdim**2]) ** (-np.min([2*a, b])*c)
    print("Regularizer", regul_dens)
    
    inp_lscale = out_lscale = 0.01
    #inp_lscale = median_heuristic(rvs[:,:1], "euclidean", False)
    out_lscale = 4*(ref_samps.T[1,1] - ref_samps.T[1,0])
    #out_lscale = np.min([4*(ref_samps.T[1,1] - ref_samps.T[1,0]), out_lscale])
    
    print("out", out_lscale, "in", inp_lscale)
    in_kern = GaussianKernel(inp_lscale)
    #in_kern = LaplaceKernel(10*inp_lscale)
    out_kern = GaussianKernel(out_lscale)
    #out_kern = LaplaceKernel(8*out_lscale)

    
    cdo = CorrectedConditionDensityOperator(rvs[:, :1], rvs[:, 1:], ref_samps, in_kern, out_kern, inp_regul=regul_dens, outp_regul=regul_dens)
    #crnvp = CondRealNVP.optimized_cde(rvs[:, :1], rvs[:, 1:], batch_size="Full Dataset")    
    
    perm = rvs[np.random.permutation(len(rvs))[:400], :].T

    x1 = 1
    x2 = 0
    (true_x1, est_x1, true_x2, est_x2) = [lambda samps: true_dens(np.hstack([x1, samps])),
                                            lambda samps: (cdo.lhood(np.ones((1,1))*x1, samps, normalize=True)),
                                            lambda samps: true_dens(np.hstack([x2, samps])),
                                            lambda samps: (cdo.lhood(np.ones((1,1))*x2, samps, normalize=True))]
    # (true_x1, est_x1, true_x2, est_x2) = [lambda samps: true_dens(np.hstack([x1, samps])),
    #                                         lambda samps: exp(crnvp.log_probs(np.ones((samps.shape[0],1))*x1, samps, )),
    #                                         lambda samps: true_dens(np.hstack([x2, samps])),
    #                                         lambda samps: exp(crnvp.log_probs(np.ones((samps.shape[0],1))*x2, samps, ))]
    if plot:
        mg, heights = compute_dens((true_x1, est_x1, true_x2, est_x2), [(mmin[0]-.3, mmax[0]+.3), (-1, +1)])
        heights = heights.squeeze()

        #plot_3d_and_cross_sections(mg, heights[2:], heights[:2], perm, [0, 1],)
        
        cont_mult(mg, np.vstack([heights[2:],heights[:2]]), titles=[r"truth at $x=0$", r"estimate at $x=0$", r"truth at $x=1$", r"estimate at $x=1$"], vminmax=True)
        scatter_3d(perm)
        return s, mg, heights, perm
    
