#  Copyright (c) 2021. Robin Thibaut, Ghent University

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
import warnings
from os.path import join as jp
from typing import List

import joblib
import numpy as np
from sklearn.cross_decomposition import CCA

import experiment._utils as fops
import experiment.processing.predictor_handle as dops
from experiment._spatial import grid_parameters, signed_distance
from experiment._visualization import whpa_plot
from experiment.processing.dimension_reduction import PC

Root = List[str]
Combination = List[int]


def base_pca(base,
             base_dir: str,
             roots: Root,
             test_roots: Root,
             d_pca_obj=None,
             h_pca_obj=None,
             check: bool = False):
    """
    Initiate BEL by performing PCA on the training targets or features.
    :param base: class: Base class object
    :param base_dir: str: Base directory path
    :param roots: list:
    :param test_roots: list:
    :param d_pca_obj:
    :param h_pca_obj:
    :param check: bool: Flag to plot
    :return:
    """
    if d_pca_obj is not None:
        # Loads the results:
        tc0, _, _ = fops.data_loader(roots=roots, d=True)
        # tc0 = breakthrough curves with shape (n_sim, n_wells, n_time_steps)
        # pzs = WHPA
        # roots_ = simulation id
        # Subdivide d in an arbitrary number of time steps:
        tc = dops.curve_interpolation(tc0=tc0)  # tc has shape (n_sim, n_wells, n_time_steps)
        # with n_sim = n_training + n_test
        # PCA on transport curves
        d_pco = PC(name='d', training=tc, roots=roots, directory=os.path.dirname(d_pca_obj))
        d_pco.training_fit_transform()
        # Dump
        joblib.dump(d_pco, d_pca_obj)

    x_lim, y_lim, grf = base.Focus.x_range, base.Focus.y_range, base.Focus.cell_dim

    if h_pca_obj is not None:
        # Loads the results:
        _, pzs, r = fops.data_loader(roots=roots, h=True)
        # Load parameters:
        xys, nrow, ncol = grid_parameters(x_lim=x_lim, y_lim=y_lim, grf=grf)  # Initiate SD instance

        # PCA on signed distance
        # Compute signed distance on pzs.
        # h is the matrix of target feature on which PCA will be performed.
        h = np.array([signed_distance(xys, nrow, ncol, grf, pp) for pp in pzs])

        if check:
            # Load parameters:
            # instance
            fig_dir = jp(os.path.dirname(h_pca_obj), 'roots_whpa')
            fops.dirmaker(fig_dir)
            for i, e in enumerate(h):
                whpa_plot(whpa=[e],
                          x_lim=x_lim,
                          y_lim=y_lim,
                          well_comb=base.Wells.combination,
                          lw=1,
                          fig_file=jp(fig_dir, ''.join((r[i], '.png'))))
                np.save(jp(fig_dir, ''.join((r[i], '.npy'))), e)

        # Initiate h pca object
        h_pco = PC(name='h', training=h, roots=roots, directory=base_dir)
        # Transform
        h_pco.training_fit_transform()
        # Define number of components to keep
        # h_pco.n_pca_components(.98)  # Number of components for signed distance automatically set.
        h_pco.n_pc_cut = base.HyperParameters.n_pc_target
        # Dump
        joblib.dump(h_pco, h_pca_obj)

        if not os.path.exists(jp(base_dir, 'roots.dat')):  # Save roots id's in a dat file
            with open(jp(base_dir, 'roots.dat'), 'w') as f:
                for r in roots:  # Saves roots name until test roots
                    f.write(os.path.basename(r) + '\n')

        if not os.path.exists(jp(base_dir, 'test_roots.dat')):  # Save roots id's in a dat file
            with open(jp(base_dir, 'test_roots.dat'), 'w') as f:
                for r in test_roots:  # Saves roots name until test roots
                    f.write(os.path.basename(r) + '\n')


def bel_fit_transform(base,
                      well_comb: Combination = None,
                      training_roots: Root = None,
                      test_root: Root = None):
    """
    This function loads raw data and perform both PCA and CCA on it.
    It saves results as pkl objects that have to be loaded in the forecast_error.py script to perform predictions.

    :param training_roots: list: List containing the uuid's of training roots
    :param base: class: Base class object containing global constants.
    :param well_comb: list: List of injection wells used to make prediction
    :param test_root: list: Folder path containing output to be predicted
    """

    # Load parameters:
    x_lim, y_lim, grf = base.Focus.x_range, base.Focus.y_range, base.Focus.cell_dim
    xys, nrow, ncol = grid_parameters(x_lim=x_lim, y_lim=y_lim, grf=grf)  # Initiate SD instance

    if well_comb is not None:
        base.Wells.combination = well_comb

    # Directories
    md = base.Directories()
    res_dir = md.hydro_res_dir  # Results folders of the hydro simulations

    # Parse test_root
    if isinstance(test_root, str):  # If only one root given
        if os.path.exists(jp(res_dir, test_root)):
            test_root = [test_root]
        else:
            warnings.warn('Specified folder {} does not exist'.format(test_root[0]))

    bel_dir = jp(md.forecasts_dir, test_root[0])  # Directory in which to load forecasts

    base_dir = jp(md.forecasts_dir, 'base')  # Base directory that will contain target objects and processed data

    new_dir = ''.join(list(map(str, base.Wells.combination)))  # sub-directory for forecasts
    sub_dir = jp(bel_dir, new_dir)

    # %% Folders
    obj_dir = jp(sub_dir, 'obj')
    fig_data_dir = jp(sub_dir, 'data')
    fig_pca_dir = jp(sub_dir, 'pca')
    fig_cca_dir = jp(sub_dir, 'cca')
    fig_pred_dir = jp(sub_dir, 'uq')

    # %% Creates directories
    [fops.dirmaker(f) for f in [obj_dir, fig_data_dir, fig_pca_dir, fig_cca_dir, fig_pred_dir]]

    # Load training data
    tsub = jp(base_dir, 'training_curves.npy')  # Refined breakthrough curves data file
    if not os.path.exists(tsub):
        # Loads the results:
        tc0, _, _ = fops.data_loader(res_dir=res_dir, roots=training_roots, d=True)
        # tc0 = breakthrough curves with shape (n_sim, n_wells, n_time_steps)
        # pzs = WHPA's
        # roots_ = simulations id's
        # Subdivide d in an arbitrary number of time steps:
        tc = dops.curve_interpolation(tc0=tc0, n_time_steps=200)  # tc has shape (n_sim, n_wells, n_time_steps)
        # with n_sim = n_training + n_test
        np.save(tsub, tc)
        # Save file roots
    else:
        tc = np.load(tsub)

    # %% Select wells:
    selection = [wc - 1 for wc in base.Wells.combination]
    tc = tc[:, selection, :]

    # %%  PCA
    # PCA is performed with maximum number of components.
    # We choose an appropriate number of components to keep later on.

    # PCA on transport curves
    d_pco = PC(name='d', training=tc, roots=training_roots, directory=obj_dir)
    d_pco.training_fit_transform()
    # PCA on transport curves
    d_pco.n_pc_cut = base.HyperParameters.n_pc_predictor
    ndo = d_pco.n_pc_cut
    n_time_steps = base.HyperParameters.n_tstp
    # Load observation (test_root)
    tc0, _, _ = fops.data_loader(res_dir=res_dir, test_roots=test_root, d=True)
    # Subdivide d in an arbitrary number of time steps:
    tcp = dops.curve_interpolation(tc0=tc0, n_time_steps=n_time_steps)
    tcp = tcp[:, selection, :]  # Extract desired observation
    d_pco.test_transform(tcp, test_root=test_root)  # Perform transformation on testing curves
    d_pc_training, _ = d_pco.comp_refresh(ndo)  # Split

    # Save the d PC object.
    joblib.dump(d_pco, jp(obj_dir, 'd_pca.pkl'))

    # PCA on signed distance from base object containing training instances
    h_pco = joblib.load(jp(base_dir, 'h_pca.pkl'))
    nho = h_pco.n_pc_cut  # Number of components to keep
    # Load whpa to predict
    _, pzs, _ = fops.data_loader(roots=test_root, h=True)
    # Compute WHPA on the prediction
    if h_pco.predict_pc is None:
        h = np.array([signed_distance(xys, nrow, ncol, grf, pp) for pp in pzs])
        # Perform PCA
        h_pco.test_transform(h, test_root=test_root)
        # Cut desired number of components
        h_pc_training, _ = h_pco.comp_refresh(nho)
        # Save updated PCA object in base
        joblib.dump(h_pco, jp(base_dir, 'h_pca.pkl'))

        fig_dir = jp(base_dir, 'roots_whpa')
        fops.dirmaker(fig_dir)
        np.save(jp(fig_dir, test_root[0]), h)  # Save the prediction WHPA
    else:
        # Cut components
        h_pc_training, _ = h_pco.comp_refresh(nho)

    # %% CCA
    n_comp_cca = min(ndo, nho)  # Number of CCA components is chosen as the min number of PC
    # components between d and h.
    # By default, it scales the data
    # TODO: Check max_iter & tol
    cca = CCA(n_components=n_comp_cca, scale=True, max_iter=500*20, tol=1e-06)
    cca.fit(X=d_pc_training, Y=h_pc_training)  # Fit
    joblib.dump(cca, jp(obj_dir, 'cca.pkl'))  # Save the fitted CCA operator

    return sub_dir
