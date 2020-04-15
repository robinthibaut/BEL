"""
This script pre-processes the data.
- It subdivides the breakthrough curves into an arbitrary number of steps, as the mt3dms results
do not necessarily share the same time steps - d
- It computes the signed distance field for each particles endpoints file - h
It then perform PCA keeping all components on both d and h.
Finally, CCA is performed after selecting an appropriate number of PC to keep.

It saves 2 pca objects (d, h) and 1 cca object, according to the project ecosystem.
"""
#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import uuid
from os.path import join as jp

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import CCA

from bel.toolbox.file_ops import FileOps
from bel.toolbox.data_ops import DataOps
from bel.toolbox.mesh_ops import MeshOps
from bel.toolbox.plots import Plot
from bel.toolbox.pca_ops import PCAOps
from bel.toolbox.posterior_ops import PosteriorOps
from bel.processing.signed_distance import SignedDistance

plt.style.use('dark_background')


do = DataOps()
mo = MeshOps()
po = PosteriorOps()
x_lim, y_lim, grf = [800, 1150], [300, 700], 2
sd = SignedDistance(x_lim=x_lim, y_lim=y_lim, grf=grf)
mp = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf)


def bel(n_training=300, n_test=5, new_dir=None):
    """
    This function loads raw data and perform both PCA and CCA on it.
    It saves results as pkl objects that have to be loaded in the forecast_error.py script to perform predictions.

    :param n_training: Number of samples used to train the model
    :param n_test: Number of samples on which to perform prediction
    :param new_dir: Name of the forecast directory
    :return:
    """
    # Directories
    res_dir = jp('..', 'hydro', 'results')  # Results folders of the hydro simulations
    bel_dir = jp('..', 'forecasts')  # Directory in which to load forecasts

    if new_dir is not None:  # If a new_dir is provided, it assumes that a roots.dat file exist in that folder.
        with open(jp(bel_dir, new_dir, 'roots.dat')) as f:
            roots = f.read().splitlines()
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
    [FileOps.dirmaker(f) for f in [obj_dir, fig_data_dir, fig_pca_dir, fig_cca_dir, fig_pred_dir]]

    n = n_training + n_test  # Total number of simulations to load, only has effect if NO roots file is loaded.
    check = False  # Flag to check for simulations issues
    tc0, pzs, roots_ = FileOps.load_res(res_dir=res_dir, n=n, check=check, roots=roots)
    # Save file roots
    with open(jp(sub_dir, 'roots.dat'), 'w') as f:
        for r in roots_:
            f.write(os.path.basename(r) + '\n')

    # Compute signed distance on pzs.
    # h is the matrix of target feature on which PCA will be performed.
    h = np.array([sd.function(pp) for pp in pzs])
    # Plot all WHPP
    mp.whp(h, fig_file=jp(fig_data_dir, 'all_whpa.png'), show=True)

    # Subdivide d in an arbitrary number of time steps.
    tc = do.d_process(tc0=tc0, n_time_steps=250)
    n_wel = len(tc[0])  # Number of injecting wels

    # Plot d
    mp.curves(tc=tc, n_wel=n_wel, sdir=fig_data_dir)
    mp.curves_i(tc=tc, n_wel=n_wel, sdir=fig_data_dir)

    # %%  PCA
    # PCA is performed with maximum number of components.
    # We choose an appropriate number of components to keep later on.

    # Choose size of training and prediction set
    n_sim = len(h)  # Number of simulations
    n_obs = n_test  # Number of 'observations' on which the predictions will be made.
    n_training = n_sim - n_obs  # number of synthetic data that will be used for constructing our prediction model

    # PCA on transport curves
    d_pco = PCAOps(name='d', raw_data=tc, directory=obj_dir)
    d_pco.pca_tp(n_training)  # Split into training and prediction
    d_pc_training, d_pc_prediction = d_pco.pca_transformation()  # Performs transformation

    # PCA on signed distance
    h_pco = PCAOps(name='h', raw_data=h, directory=obj_dir)
    h_pco.pca_tp(n_training) # Split into training and prediction
    h_pc_training, h_pc_prediction = h_pco.pca_transformation()

    # TODO: Build a framework to select the number of PC components.
    # Choose number of PCA components to keep.
    # Compares true value with inverse transformation from PCA
    ndo = d_pco.n_pca_components(.999)  # Number of components for breakthrough curves
    nho = h_pco.n_pca_components(.98)  # Number of components for signed distance

    # Explained variance plots
    mp.explained_variance(d_pco.operator, n_comp=ndo, fig_file=jp(fig_pca_dir, 'd_exvar.png'), show=True)
    mp.explained_variance(h_pco.operator, n_comp=nho, fig_file=jp(fig_pca_dir, 'h_exvar.png'), show=True)

    # Scores plots
    mp.pca_scores(d_pc_training, d_pc_prediction, n_comp=ndo, fig_file=jp(fig_pca_dir, 'd_scores.png'), show=True)
    mp.pca_scores(h_pc_training, h_pc_prediction, n_comp=nho, fig_file=jp(fig_pca_dir, 'h_scores.png'), show=True)

    # Assign final n_comp for PCA
    n_d_pc_comp = ndo
    n_h_pc_comp = nho

    # Cut desired number of PC components
    d_pc_training, d_pc_prediction = d_pco.pca_refresh(n_d_pc_comp)
    h_pc_training, h_pc_prediction = h_pco.pca_refresh(n_h_pc_comp)

    # Save the d and h PC objects.
    joblib.dump(d_pco, jp(obj_dir, 'd_pca.pkl'))
    joblib.dump(h_pco, jp(obj_dir, 'h_pca.pkl'))
    # %% CCA

    n_comp_cca = min(n_d_pc_comp, n_h_pc_comp)  # Number of CCA components is chosen as the min number of PC
    # components between d and h.
    float_epsilon = np.finfo(float).eps
    # By default, it scales the data
    cca = CCA(n_components=n_comp_cca, scale=True, max_iter=int(500 * 20), tol=float_epsilon * 10)
    cca.fit(d_pc_training, h_pc_training)  # Fit
    joblib.dump(cca, jp(obj_dir, 'cca.pkl'))  # Save the fitted CCA operator


if __name__ == "__main__":
    bel(new_dir=None)

