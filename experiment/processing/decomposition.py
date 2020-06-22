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
from experiment.math.signed_distance import SignedDistance
from experiment.processing.pca import PCAIO


def base_pca(base, roots, d_pca_obj=None, h_pca_obj=None, check=False):
    if d_pca_obj is not None:
        # Loads the results:
        tc0, _, _ = fops.load_res(roots=roots, d=True)
        # tc0 = breakthrough curves with shape (n_sim, n_wels, n_time_steps)
        # pzs = WHPA
        # roots_ = simulation id
        # Subdivide d in an arbitrary number of time steps:
        tc = dops.d_process(tc0=tc0, n_time_steps=200)  # tc has shape (n_sim, n_wels, n_time_steps)
        # with n_sim = n_training + n_test
        # PCA on transport curves
        d_pco = PCAIO(name='d', training=tc, roots=roots, directory=os.path.dirname(d_pca_obj))
        d_pco.pca_training_transformation()
        d_pco.n_pca_components(.999)  # Number of components for breakthrough curves
        # Dump
        joblib.dump(d_pco, d_pca_obj)

    x_lim, y_lim, grf = base.Focus.x_range, base.Focus.y_range, base.Focus.cell_dim

    if h_pca_obj is not None:
        # Loads the results:
        _, pzs, r = fops.load_res(roots=roots, h=True)
        # Load parameters:
        sd = SignedDistance(x_lim=x_lim, y_lim=y_lim, grf=grf)  # Initiate SD instance

        # PCA on signed distance
        # Compute signed distance on pzs.
        # h is the matrix of target feature on which PCA will be performed.
        h = np.array([sd.compute(pp) for pp in pzs])

        if check:
            # Load parameters:
            mp = plot.Plot(x_lim=x_lim, y_lim=y_lim, grf=grf, wel_comb=base.Wels.combination)  # Initiate Plot instance
            fig_dir = jp(os.path.dirname(h_pca_obj), 'roots_whpa')
            fops.dirmaker(fig_dir)
            for i, e in enumerate(h):
                mp.whp([e],
                       lw=1,
                       fig_file=jp(fig_dir, ''.join((r[i], '.png'))))
                np.save(jp(fig_dir, ''.join((r[i], '.npy'))), e)

            # return

        # Initiate h pca object
        h_pco = PCAIO(name='h', training=h, roots=roots, directory=os.path.dirname(h_pca_obj))
        # Transform
        h_pco.pca_training_transformation()
        # Define number of components to keep
        h_pco.n_pca_components(.98)  # Number of components for signed distance
        # Dump
        joblib.dump(h_pco, h_pca_obj)


def bel(base, wel_comb=None, training_roots=None, test_roots=None, **kwargs):
    """
    This function loads raw data and perform both PCA and CCA on it.
    It saves results as pkl objects that have to be loaded in the forecast_error.py script to perform predictions.

    :param wel_comb: List of injection wels used to make prediction
    :param test_roots: Folder paths containing outputs to be predicted

    """

    # Load parameters:
    x_lim, y_lim, grf = base.Focus.x_range, base.Focus.y_range, base.Focus.cell_dim
    sd = SignedDistance(x_lim=x_lim, y_lim=y_lim, grf=grf)  # Initiate SD instance

    if wel_comb is not None:
        base.Wels.combination = wel_comb

    mp = plot.Plot(x_lim=x_lim, y_lim=y_lim, grf=grf, wel_comb=base.Wels.combination)  # Initiate Plot instance

    # Directories
    md = base.Directories()
    res_dir = md.hydro_res_dir  # Results folders of the hydro simulations

    # Parse test_roots
    if isinstance(test_roots, str):  # If only one root given
        if os.path.exists(jp(res_dir, test_roots)):
            test_roots = [test_roots]
        else:
            warnings.warn('Specified folder {} does not exist'.format(test_roots[0]))

    bel_dir = jp(md.forecasts_dir, test_roots[0])  # Directory in which to load forecasts

    base_dir = jp(md.forecasts_dir, 'base')  # Base directory that will contain target objects and processed data

    new_dir = ''.join(list(map(str, base.Wels.combination)))  # sub-directory for forecasts
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

    # Load training data
    tsub = jp(base_dir, 'training_curves.npy')  # Refined breakthrough curves data file
    if not os.path.exists(tsub):
        # Loads the results:
        tc0, _, _ = fops.load_res(res_dir=res_dir, roots=training_roots, d=True)
        # tc0 = breakthrough curves with shape (n_sim, n_wels, n_time_steps)
        # pzs = WHPA
        # roots_ = simulation id
        # Subdivide d in an arbitrary number of time steps:
        tc = dops.d_process(tc0=tc0, n_time_steps=200)  # tc has shape (n_sim, n_wels, n_time_steps)
        # with n_sim = n_training + n_test
        np.save(tsub, tc)
        # Save file roots
    else:
        tc = np.load(tsub)

    if not os.path.exists(jp(base_dir, 'roots.dat')):
        with open(jp(base_dir, 'roots.dat'), 'w') as f:
            for r in training_roots:  # Saves roots name until test roots
                f.write(os.path.basename(r) + '\n')

    # %% Select wels:
    selection = [wc - 1 for wc in base.Wels.combination]
    tc = tc[:, selection, :]

    # %%  PCA
    # PCA is performed with maximum number of components.
    # We choose an appropriate number of components to keep later on.

    # PCA on transport curves
    d_pco = PCAIO(name='d', training=tc, roots=training_roots, directory=obj_dir)
    d_pco.pca_training_transformation()
    # d_pco.n_pca_components(.999)  # Number of components for breakthrough curves
    # PCA on transport curves
    d_pco.ncomp = 50
    ndo = d_pco.ncomp
    # Load test
    tc0, _, _ = fops.load_res(res_dir=res_dir, test_roots=test_roots, d=True)
    # Subdivide d in an arbitrary number of time steps:
    tcp = dops.d_process(tc0=tc0, n_time_steps=200)
    tcp = tcp[:, selection, :]
    d_pco.pca_test_transformation(tcp, test_roots=test_roots)  # Performs transformation on testing curves
    d_pc_training, d_pc_prediction = d_pco.pca_refresh(ndo)  # Performs transformation on training curves
    # Save the d PC object.
    joblib.dump(d_pco, jp(obj_dir, 'd_pca.pkl'))
    # Plot d:
    mp.curves(tc=np.concatenate((tc, tcp), axis=0), sdir=fig_data_dir, highlight=[len(tc)])
    mp.curves_i(tc=np.concatenate((tc, tcp), axis=0), sdir=fig_data_dir, highlight=[len(tc)])

    # PCA on signed distance
    h_pco = joblib.load(jp(base_dir, 'h_pca.pkl'))
    nho = h_pco.ncomp  # Number of components to keep
    # Load
    _, pzs, _ = fops.load_res(roots=test_roots, h=True)
    # Compute WHPA
    if h_pco.predict_pc is None:
        h = np.array([sd.compute(pp) for pp in pzs])
        h_pco.pca_test_transformation(h, test_roots=test_roots)
        h_pc_training, h_pc_prediction = h_pco.pca_refresh(nho)
        joblib.dump(h_pco, jp(base_dir, 'h_pca.pkl'))

        fig_dir = jp(base_dir, 'roots_whpa')
        fops.dirmaker(fig_dir)
        np.save(jp(fig_dir, test_roots[0]), h)  # Save the prediction WHPA
        ff = jp(fig_dir, f'{test_roots[0]}.png')
        h_training = h_pco.training_physical.reshape(h_pco.shape)
        mp.whp(h_training, alpha=.2, show=False)
        mp.whp(h, colors='r', lw=1, alpha=1, fig_file=ff)

    else:
        # Cut components
        h_pc_training, h_pc_prediction = h_pco.pca_refresh(nho)


    # %% CCA

    n_comp_cca = min(ndo, nho)  # Number of CCA components is chosen as the min number of PC
    # components between d and h.
    float_epsilon = np.finfo(float).eps
    # By default, it scales the data
    cca = CCA(n_components=n_comp_cca, scale=True, max_iter=int(500 * 20), tol=float_epsilon * 10)
    cca.fit(d_pc_training, h_pc_training)  # Fit
    joblib.dump(cca, jp(obj_dir, 'cca.pkl'))  # Save the fitted CCA operator

    return sub_dir
