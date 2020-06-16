#  Copyright (c) 2020. Robin Thibaut, Ghent University

"""
This script pre-processes the data.

- It subdivides the breakthrough curves into an arbitrary number of steps, as the mt3dms results
do not necessarily share the same time steps - d

- It computes the signed distance field for each particles endpoints file - h
It then perform PCA keeping all components on both d and h.

- Finally, CCA is performed after selecting an appropriate number of PC to keep.

It saves 2 pca objects (d, h) and 1 cca object, according to the project ecosystem.
"""

import os
import uuid
import warnings
from os.path import join as jp

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import CCA

import experiment.goggles.visualization as plot
import experiment.processing.examples as dops
import experiment.toolbox.filesio as fops
from experiment.base.inventory import Directories, Focus, Wels
from experiment.math.signed_distance import SignedDistance
from experiment.processing.pca import PCAIO


def roots_pca(roots, h_pca_obj, n_test=1):

    # Load parameters:
    x_lim, y_lim, grf = Focus.x_range, Focus.y_range, Focus.cell_dim
    sd = SignedDistance(x_lim=x_lim, y_lim=y_lim, grf=grf)  # Initiate SD instance

    # Loads the results:
    _, pzs, roots_ = fops.load_res(roots=roots)

    # %%  PCA
    # PCA is performed with maximum number of components.
    # We choose an appropriate number of components to keep later on.

    # Choose size of training and prediction set
    n_sim = len(pzs)  # Number of simulations
    n_obs = n_test  # Number of 'observations' on which the predictions will be made.
    n_training = n_sim - n_obs  # number of synthetic data that will be used for constructing our prediction model

    # PCA on signed distance
    # Compute signed distance on pzs.
    # h is the matrix of target feature on which PCA will be performed.
    h = np.array([sd.compute(pp) for pp in pzs])
    # Plot all WHPP
    # Initiate h pca object
    h_pco = PCAIO(name='h', raw_data=h, directory=os.path.dirname(h_pca_obj))
    # Split into training and prediction
    h_pco.pca_tp(n_training)
    # Transform
    h_pco.pca_transformation()
    # Define number of components to keep
    nho = h_pco.n_pca_components(.98)  # Number of components for signed distance
    # Split
    h_pc_training, h_pc_prediction = h_pco.pca_refresh(nho)
    # Dump
    joblib.dump(h_pco, h_pca_obj)


def bel(n_training=200, n_test=1, wel_comb=None, training_roots=None, test_roots=None, target_pca=None):
    """
    This function loads raw data and perform both PCA and CCA on it.
    It saves results as pkl objects that have to be loaded in the forecast_error.py script to perform predictions.

    :param wel_comb: List of injection wels used to make prediction
    :param test_roots: Folder paths containing outputs to be predicted
    :param n_training: Number of samples used to train the model
    :param n_test: Number of samples on which to perform prediction
    :param base: Flag for base folder

    """

    # Load parameters:
    x_lim, y_lim, grf = Focus.x_range, Focus.y_range, Focus.cell_dim
    sd = SignedDistance(x_lim=x_lim, y_lim=y_lim, grf=grf)  # Initiate SD instance

    n_sim = n_training + n_test  # Total number of simulations to load, only has effect if NO roots file is loaded.

    if wel_comb is not None:
        Wels.combination = wel_comb

    mp = plot.Plot(x_lim=x_lim, y_lim=y_lim, grf=grf, wel_comb=Wels.combination)  # Initiate Plot instance

    # Directories
    md = Directories()
    res_dir = md.hydro_res_dir  # Results folders of the hydro simulations

    # Parse test_roots
    if isinstance(test_roots, str):  # If only one root given
        if os.path.exists(jp(res_dir, test_roots)):
            test_roots = [test_roots]
            n_test = 1
        else:
            warnings.warn('Specified folder {} does not exist'.format(test_roots[0]))

    bel_dir = jp(md.forecasts_dir, test_roots[0])  # Directory in which to load forecasts

    base_dir = jp(md.forecasts_dir, 'base')  # Base directory that will contain target objects and processed data

    new_dir = ''.join(list(map(str, Wels.combination)))  # sub-directory for forecasts
    sub_dir = jp(bel_dir, new_dir)

    # %% Folders
    obj_dir = jp(sub_dir, 'obj')
    fig_data_dir = jp(sub_dir, 'data')
    fig_pca_dir = jp(sub_dir, 'pca')
    fig_cca_dir = jp(sub_dir, 'cca')
    fig_pred_dir = jp(sub_dir, 'uq')
    # TODO: pass them to next class

    # %% Creates directories

    [fops.dirmaker(f) for f in [obj_dir, fig_data_dir, fig_pca_dir, fig_cca_dir, fig_pred_dir]]

    tsub = jp(obj_dir, 'tc.npy')  # Refined breakthrough curves data file
    if not os.path.exists(tsub):
        # Loads the results:
        tc0, pzs, roots_ = fops.load_res(res_dir=res_dir,
                                         roots=training_roots,
                                         test_roots=test_roots)
        # tc0 = breakthrough curves with shape (n_sim, n_wels, n_time_steps)
        # pzs = WHPA
        # roots_ = simulation id

        # Subdivide d in an arbitrary number of time steps:
        tc = dops.d_process(tc0=tc0, n_time_steps=200)  # tc has shape (n_sim, n_wels, n_time_steps)

        # with n_sim = n_training + n_test
        np.save(tsub, tc)

        # Save file roots
        if not os.path.exists(jp(base_dir, 'roots.dat')):
            with open(jp(base_dir, 'roots.dat'), 'w') as f:
                for r in roots_[:-n_test]:  # Saves roots name until test roots
                    f.write(os.path.basename(r) + '\n')

    else:
        tc = np.load(tsub)

    # %% Select wels:
    selection = [wc - 1 for wc in Wels.combination]
    tc = tc[:, selection, :]

    # Plot d:
    mp.curves(tc=tc, sdir=fig_data_dir, highlight=[n_sim-1])
    mp.curves_i(tc=tc, sdir=fig_data_dir, highlight=[n_sim-1])

    # %%  PCA
    # PCA is performed with maximum number of components.
    # We choose an appropriate number of components to keep later on.

    # Choose size of training and prediction set
    n_sim = len(tc)  # Number of simulations
    n_obs = n_test  # Number of 'observations' on which the predictions will be made.
    n_training = n_sim - n_obs  # number of synthetic data that will be used for constructing our prediction model

    # PCA on transport curves
    d_pco = PCAIO(name='d', raw_data=tc, directory=obj_dir)
    d_pco.pca_tp(n_training)  # Split into training and prediction
    d_pc_training, d_pc_prediction = d_pco.pca_transformation()  # Performs transformation

    # PCA on signed distance
    if target_pca is None:
        if not os.path.exists(jp(base_dir, 'h_pca.pkl')):
            # Compute signed distance on pzs.
            # h is the matrix of target feature on which PCA will be performed.
            h = np.array([sd.compute(pp) for pp in pzs])
            # Plot all WHPP
            # mp.whp(h, fig_file=jp(fig_data_dir, 'all_whpa.png'), show=False)
            mp.whp_prediction(forecasts=h,
                              h_true=h[-1],
                              h_pred=None,
                              show_wells=True,
                              fig_file=jp(fig_data_dir, 'all_whpa.png'))
            # Initiate h pca object
            h_pco = PCAIO(name='h', raw_data=h, directory=obj_dir)
            # Split into training and prediction
            h_pco.pca_tp(n_training)
            # Transform
            h_pco.pca_transformation()
            # Define number of components to keep
            nho = h_pco.n_pca_components(.98)  # Number of components for signed distance
            # Split
            h_pc_training, h_pc_prediction = h_pco.pca_refresh(nho)
            # Dump
            joblib.dump(h_pco, jp(base_dir, 'h_pca.pkl'))
            # Plot
            plot.explained_variance(h_pco.operator,
                                    n_comp=nho,
                                    fig_file=jp(fig_pca_dir, 'h_exvar.png'), show=True)
            plot.pca_scores(h_pc_training,
                            h_pc_prediction,
                            n_comp=nho,
                            fig_file=jp(fig_pca_dir, 'h_scores.png'), show=True)
        else:
            h_pco = joblib.load(jp(base_dir, 'h_pca.pkl'))
            nho = h_pco.ncomp
            h_pc_training, h_pc_prediction = h_pco.pca_refresh(nho)
    else:
        h_pco = joblib.load(target_pca)
        nho = h_pco.ncomp
        h_pc_training, h_pc_prediction = h_pco.pca_refresh(nho)

    # TODO: Build a framework to select the number of PC components.
    # Choose number of PCA components to keep.
    # Compares true value with inverse transformation from PCA
    ndo = d_pco.n_pca_components(.999)  # Number of components for breakthrough curves

    # Explained variance plots
    plot.explained_variance(d_pco.operator, n_comp=ndo, fig_file=jp(fig_pca_dir, 'd_exvar.png'), show=True)

    # Scores plots
    plot.pca_scores(d_pc_training, d_pc_prediction, n_comp=ndo, fig_file=jp(fig_pca_dir, 'd_scores.png'), show=True)

    # Assign final n_comp for PCA
    n_d_pc_comp = ndo
    n_h_pc_comp = nho

    # Cut desired number of PC components
    d_pc_training, d_pc_prediction = d_pco.pca_refresh(n_d_pc_comp)

    # Save the d PC object.
    joblib.dump(d_pco, jp(obj_dir, 'd_pca.pkl'))

    # %% CCA

    n_comp_cca = min(n_d_pc_comp, n_h_pc_comp)  # Number of CCA components is chosen as the min number of PC
    # components between d and h.
    float_epsilon = np.finfo(float).eps
    # By default, it scales the data
    cca = CCA(n_components=n_comp_cca, scale=True, max_iter=int(500 * 20), tol=float_epsilon * 10)
    cca.fit(d_pc_training, h_pc_training)  # Fit
    joblib.dump(cca, jp(obj_dir, 'cca.pkl'))  # Save the fitted CCA operator

    return sub_dir

