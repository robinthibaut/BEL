#  Copyright (c) 2020. Robin Thibaut, Ghent University

"""
This script pre-processes the data.
- It subdivides the breakthrough curves into an arbitrary number of steps, as the mt3dms results
do not necessarily share the same time steps - d
- It computes the signed distance field for each particles endpoints file - h
It then perform PCA keeping all components on both d and h.
Finally, CCA is performed after selecting an appropriate number of PC to keep.

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


def bel(n_training=300, n_test=5, wel_comb=None, new_dir=None, base=None, test_roots=None):
    """
    This function loads raw data and perform both PCA and CCA on it.
    It saves results as pkl objects that have to be loaded in the forecast_error.py script to perform predictions.

    :param wel_comb: List of injection wels used to make prediction
    :param test_roots: Folder paths containing outputs to be predicted
    :param n_training: Number of samples used to train the model
    :param n_test: Number of samples on which to perform prediction
    :param new_dir: Name of the forecast directory

    """

    # Load parameters:
    fc = Focus()
    x_lim, y_lim, grf = fc.x_range, fc.y_range, fc.cell_dim
    sd = SignedDistance(x_lim=x_lim, y_lim=y_lim, grf=grf)  # Initiate SD instance

    # Wels data
    wels = Wels()
    if wel_comb is not None:
        wels.combination = wel_comb

    mp = plot.Plot(x_lim=x_lim, y_lim=y_lim, grf=grf, wel_comb=wels.combination)  # Initiate Plot instance

    # Directories
    md = Directories()
    res_dir = md.hydro_res_dir  # Results folders of the hydro simulations
    if base is not None:
        bel_dir = jp(md.forecasts_dir, new_dir)  # Directory in which to load forecasts
        base_dir = jp(bel_dir, 'base')
    else:
        bel_dir = md.forecasts_dir

    # Parse test_roots
    if isinstance(test_roots, (list, tuple)):
        n_test = len(test_roots)
        for f in test_roots:
            if not os.path.exists(jp(res_dir, f)):
                warnings.warn('Specified folder {} does not exist'.format(jp(res_dir, f)))
    if isinstance(test_roots, str):
        if os.path.exists(jp(res_dir, test_roots)):
            test_roots = [test_roots]
            n_test = 1
        else:
            warnings.warn('Specified folder {} does not exist'.format(test_roots[0]))

    if base is not None:
        try:
            with open(jp(base_dir, 'roots.dat')) as f:
                roots = f.read().splitlines()
            if n_test == 1:  # if only one root is studied
                new_dir = ''.join(list(map(str, wels.combination)))  # sub-directory for forecasts
        except FileNotFoundError:
            if n_test == 1:  # if only one root is studied
                new_dir = 'base'  # sub-directory for forecasts
            # roots = None
    else:  # Otherwise we start from 0.
        new_dir = str(uuid.uuid4())  # sub-directory for forecasts
        roots = None

    sub_dir = jp(bel_dir, new_dir)
    obj_dir = jp(sub_dir, 'objects')

    fig_dir = jp(sub_dir, 'figures')
    fig_data_dir = jp(fig_dir, 'Data')
    fig_pca_dir = jp(fig_dir, 'PCA')
    fig_cca_dir = jp(fig_dir, 'CCA')
    fig_pred_dir = jp(fig_dir, 'Predictions')

    # Creates directories
    [fops.dirmaker(f) for f in [obj_dir, fig_data_dir, fig_pca_dir, fig_cca_dir, fig_pred_dir]]

    n = n_training + n_test  # Total number of simulations to load, only has effect if NO roots file is loaded.
    tc0, pzs, roots_ = fops.load_res(res_dir=res_dir, n=n, roots=roots, test_roots=test_roots)  # Loads the results

    # tc0 = breakthrough curves with shape (n_sim, n_wels, n_time_steps)
    # pzs = WHPA
    # roots_ = simulation id

    # Save file roots
    if base is not None and not os.path.exists(jp(base_dir, 'roots.dat')):
        with open(jp(base_dir, 'roots.dat'), 'w') as f:
            for r in roots_:
                f.write(os.path.basename(r) + '\n')
    else:
        with open(jp(sub_dir, 'roots.dat'), 'w') as f:
            for r in roots_:
                f.write(os.path.basename(r) + '\n')

    # Subdivide d in an arbitrary number of time steps.
    tc = dops.d_process(tc0=tc0, n_time_steps=250)  # tc has shape (n_sim, n_wels, n_time_steps),
    # with n_sim = n_training + n_test
    tc = tc[:, wels.combination, :]  # Select wels
    # Plot d
    mp.curves(tc=tc, sdir=fig_data_dir)
    mp.curves_i(tc=tc, sdir=fig_data_dir)

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
    if not os.path.exists(jp(base_dir, 'objects', 'h_pca.pkl')):
        # Compute signed distance on pzs.
        # h is the matrix of target feature on which PCA will be performed.
        h = np.array([sd.compute(pp) for pp in pzs])
        # Plot all WHPP
        mp.whp(h, fig_file=jp(fig_data_dir, 'all_whpa.png'), show=False)
        h_pco = PCAIO(name='h', raw_data=h, directory=obj_dir)
        h_pco.pca_tp(n_training)  # Split into training and prediction
        h_pc_training, h_pc_prediction = h_pco.pca_transformation()
        joblib.dump(h_pco, jp(base_dir, 'h_pca.pkl'))
    else:
        h_pco = joblib.load(jp(base_dir, 'objects', 'h_pca.pkl'))
        h_pc_training, h_pc_prediction = h_pco.pca_transformation()

    # TODO: Build a framework to select the number of PC components.
    # Choose number of PCA components to keep.
    # Compares true value with inverse transformation from PCA
    ndo = d_pco.n_pca_components(.999)  # Number of components for breakthrough curves
    nho = h_pco.n_pca_components(.98)  # Number of components for signed distance

    # Explained variance plots
    plot.explained_variance(d_pco.operator, n_comp=ndo, fig_file=jp(fig_pca_dir, 'd_exvar.png'), show=True)
    plot.explained_variance(h_pco.operator, n_comp=nho, fig_file=jp(fig_pca_dir, 'h_exvar.png'), show=True)

    # Scores plots
    plot.pca_scores(d_pc_training, d_pc_prediction, n_comp=ndo, fig_file=jp(fig_pca_dir, 'd_scores.png'), show=True)
    plot.pca_scores(h_pc_training, h_pc_prediction, n_comp=nho, fig_file=jp(fig_pca_dir, 'h_scores.png'), show=True)

    # Assign final n_comp for PCA
    n_d_pc_comp = ndo
    n_h_pc_comp = nho

    # Cut desired number of PC components
    d_pc_training, d_pc_prediction = d_pco.pca_refresh(n_d_pc_comp)
    h_pc_training, h_pc_prediction = h_pco.pca_refresh(n_h_pc_comp)

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

    return new_dir

