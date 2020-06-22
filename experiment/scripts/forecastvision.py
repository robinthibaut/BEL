#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import joblib
import numpy as np
from experiment.toolbox import utils
from experiment.toolbox.filesio import load_res, folder_reset
from experiment.goggles.visualization import Plot, cca_plot, pca_scores, explained_variance
from experiment.base.inventory import MySetup


def empty_figs(res_dir):
    """ Empties figure folders """
    subdir = os.path.join(MySetup.Directories.forecasts_dir, res_dir)
    listme = os.listdir(subdir)
    folders = list(filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))

    for f in folders:
        # pca
        folder_reset(os.path.join(subdir, f, 'pca'))

        # cca
        folder_reset(os.path.join(subdir, f, 'cca'))


def pca_vision(res_dir, d=True, h=False):
    """ Loads PCA pickles and plot scores for all folders """
    subdir = os.path.join(MySetup.Directories.forecasts_dir, res_dir)
    listme = os.listdir(subdir)
    folders = list(filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))
    if d:
        for f in folders:
            dfig = os.path.join(subdir, f, 'pca')
            # For d only
            pcaf = os.path.join(subdir, f, 'obj', 'd_pca.pkl')
            d_pco = joblib.load(pcaf)
            fig_file = os.path.join(dfig, 'd_scores.png')
            pca_scores(training=d_pco.training_pc, prediction=d_pco.predict_pc, n_comp=d_pco.ncomp, fig_file=fig_file)
            # Explained variance plots
            fig_file = os.path.join(dfig, 'd_exvar.png')
            explained_variance(d_pco.operator, n_comp=d_pco.ncomp, fig_file=fig_file)
    if h:
        hbase = os.path.join(MySetup.Directories.forecasts_dir, 'base')
        # Load h pickle
        pcaf = os.path.join(hbase, 'h_pca.pkl')
        h_pco = joblib.load(pcaf)
        # Load npy whpa prediction
        prediction = np.load(os.path.join(hbase, 'roots_whpa', f'{res_dir}.npy'))
        # Transform and split
        h_pco.pca_test_transformation(prediction)
        nho = h_pco.ncomp
        h_pc_training, h_pc_prediction = h_pco.pca_refresh(nho)
        # Plot
        fig_file = os.path.join(hbase, 'roots_whpa', 'h_scores.png')
        pca_scores(training=h_pc_training, prediction=h_pc_prediction, n_comp=nho, fig_file=fig_file)
        # Explained variance plots
        fig_file = os.path.join(hbase, 'roots_whpa', 'h_exvar.png')
        explained_variance(h_pco.operator, n_comp=h_pco.ncomp, fig_file=fig_file)


def cca_vision(root_dir):
    """Loads CCA pickles and plots components for all folders"""

    subdir = os.path.join(MySetup.Directories.forecasts_dir, root_dir)
    listme = os.listdir(subdir)
    folders = list(filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))

    base_dir = os.path.join(MySetup.Directories.forecasts_dir, 'base')

    for f in folders:
        res_dir = os.path.join(subdir, f, 'obj')
        # Load objects
        f_names = list(map(lambda fn: os.path.join(res_dir, fn + '.pkl'), ['cca', 'd_pca']))
        cca_operator, d_pco = list(map(joblib.load, f_names))
        h_pco = joblib.load(os.path.join(base_dir, 'h_pca.pkl'))

        h_pred = np.load(os.path.join(base_dir, 'roots_whpa', f'{root_dir}.npy'))

        # Inspect transformation between physical and PC space
        dnc0 = d_pco.ncomp
        hnc0 = h_pco.ncomp

        # Cut desired number of PC components
        d_pc_training, d_pc_prediction = d_pco.pca_refresh(dnc0)
        h_pco.pca_test_transformation(h_pred)
        h_pc_training, h_pc_prediction = h_pco.pca_refresh(hnc0)

        # CCA plots
        d_cca_training, h_cca_training = cca_operator.transform(d_pc_training, h_pc_training)
        d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T

        cca_plot(cca_operator, d_cca_training, h_cca_training, d_pc_prediction, h_pc_prediction,
                 sdir=os.path.join(os.path.dirname(res_dir), 'cca'))


def plot_whpa():
    """Loads target pickle and plots all training WHPA"""
    x_lim, y_lim, grf = MySetup.Focus.x_range, MySetup.Focus.y_range, MySetup.Focus.cell_dim
    mplot = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf)

    fobj = os.path.join(MySetup.Directories.forecasts_dir, 'base', 'h_pca.pkl')
    h = joblib.load(fobj)
    h_training = h.training_physical.reshape(h.shape)

    mplot.whp(h_training, show=True,
              fig_file=os.path.join(MySetup.Directories.forecasts_dir, 'base', 'whpa_training.png'))


def plot_pc_ba(root):
    x_lim, y_lim, grf = MySetup.Focus.x_range, MySetup.Focus.y_range, MySetup.Focus.cell_dim
    mplot = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf)

    base_dir = os.path.join(MySetup.Directories.forecasts_dir, 'base')
    fobj = os.path.join(base_dir, 'h_pca.pkl')
    # h_pco = joblib.load(fobj)
    # hnc0 = h_pco.ncomp
    # # mplot.h_pca_inverse_plot(h_pco, hnc0, training=True, fig_dir=os.path.join(base_dir, 'control'))
    #
    # h_pred = np.load(os.path.join(base_dir, 'roots_whpa', f'{root}.npy'))
    # # Cut desired number of PC components
    # h_pco.pca_test_transformation(h_pred, test_roots=[root])
    # h_pco.pca_refresh(hnc0)
    # mplot.h_pca_inverse_plot(h_pco, hnc0, training=False, fig_dir=os.path.join(base_dir, 'control'))

    # d
    subdir = os.path.join(MySetup.Directories.forecasts_dir, root)
    listme = os.listdir(subdir)
    folders = list(filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))

    for f in folders:
        res_dir = os.path.join(subdir, f, 'obj')
        # Load objects
        d_pco = joblib.load(os.path.join(res_dir, 'd_pca.pkl'))
        dnc0 = d_pco.ncomp
        d_pco.pca_refresh(dnc0)
        setattr(d_pco, 'test_roots', [root])
        mplot.d_pca_inverse_plot(d_pco, dnc0, training=True,
                                 fig_dir=os.path.join(os.path.dirname(res_dir), 'pca'))
        mplot.d_pca_inverse_plot(d_pco, dnc0, training=False,
                                 fig_dir=os.path.join(os.path.dirname(res_dir), 'pca'))


if __name__ == '__main__':
    plot_pc_ba('6623dd4fb5014a978d59b9acb03946d2')
    # empty_figs('6623dd4fb5014a978d59b9acb03946d2')
    # cca_vision('6623dd4fb5014a978d59b9acb03946d2')
    # pca_vision('6623dd4fb5014a978d59b9acb03946d2', d=True, h=True)


