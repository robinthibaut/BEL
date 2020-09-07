#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from experiment.toolbox import utils
from experiment.toolbox.filesio import datread, load_res, folder_reset
from experiment.goggles.visualization import Plot, cca_plot, pca_scores, explained_variance
from experiment.base.inventory import MySetup


def empty_figs(root):
    """ Empties figure folders """

    if isinstance(root, (list, tuple)):
        if len(root) > 1:
            print('Input error')
            return
        else:
            root = root[0]

    subdir = os.path.join(MySetup.Directories.forecasts_dir, root)
    listme = os.listdir(subdir)
    folders = list(filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))

    for f in folders:
        # pca
        folder_reset(os.path.join(subdir, f, 'pca'))

        # cca
        folder_reset(os.path.join(subdir, f, 'cca'))


def pca_vision(root, d=True, h=False, scores=True, exvar=True, folders=None):
    """
    Loads PCA pickles and plot scores for all folders
    :param root: str:
    :param d: bool:
    :param h: bool:
    :param scores: bool:
    :param exvar: bool:
    :param folders: list:
    :return:
    """

    if isinstance(root, (list, tuple)):
        if len(root) > 1:
            print('Input error')
            return
        else:
            root = root[0]

    subdir = os.path.join(MySetup.Directories.forecasts_dir, root)
    if folders is None:
        listme = os.listdir(subdir)
        folders = list(filter(lambda du: os.path.isdir(os.path.join(subdir, du)), listme))
    else:
        if not isinstance(folders, (list, tuple)):
            folders = [folders]

    if d:
        for f in folders:
            dfig = os.path.join(subdir, f, 'pca')
            # For d only
            pcaf = os.path.join(subdir, f, 'obj', 'd_pca.pkl')
            d_pco = joblib.load(pcaf)
            fig_file = os.path.join(dfig, 'd_scores.png')
            if scores:
                pca_scores(training=d_pco.training_pc,
                           prediction=d_pco.predict_pc,
                           n_comp=d_pco.ncomp,
                           labels=False,
                           fig_file=fig_file)
            # Explained variance plots
            if exvar:
                fig_file = os.path.join(dfig, 'd_exvar.png')
                explained_variance(d_pco.operator, n_comp=d_pco.ncomp, thr=.9, fig_file=fig_file)
    if h:
        hbase = os.path.join(MySetup.Directories.forecasts_dir, 'base')
        # Load h pickle
        pcaf = os.path.join(hbase, 'h_pca.pkl')
        h_pco = joblib.load(pcaf)
        # Load npy whpa prediction
        prediction = np.load(os.path.join(hbase, 'roots_whpa', f'{root}.npy'))
        # Transform and split
        h_pco.pca_test_transformation(prediction, test_root=[root])
        nho = h_pco.ncomp
        h_pc_training, h_pc_prediction = h_pco.pca_refresh(nho)
        # Plot
        fig_file = os.path.join(hbase, 'roots_whpa', 'h_scores.png')
        if scores:
            pca_scores(training=h_pc_training,
                       prediction=h_pc_prediction,
                       n_comp=nho,
                       labels=False,
                       fig_file=fig_file)
        # Explained variance plots
        if exvar:
            fig_file = os.path.join(hbase, 'roots_whpa', 'h_exvar.png')
            explained_variance(h_pco.operator, n_comp=h_pco.ncomp, thr=.85, fig_file=fig_file)


def cca_vision(root, folders=None):
    """
    Loads CCA pickles and plots components for all folders
    :param root:
    :param folders:
    :return:
    """

    if isinstance(root, (list, tuple)):
        if len(root) > 1:
            print('Input error')
            return
        else:
            root = root[0]

    subdir = os.path.join(MySetup.Directories.forecasts_dir, root)

    if folders is None:
        listme = os.listdir(subdir)
        folders = list(filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))
    else:
        if not isinstance(folders, (list, tuple)):
            folders = [folders]
        else:
            folders = folders

    base_dir = os.path.join(MySetup.Directories.forecasts_dir, 'base')

    for f in folders:
        res_dir = os.path.join(subdir, f, 'obj')
        # Load objects
        f_names = list(map(lambda fn: os.path.join(res_dir, fn + '.pkl'), ['cca', 'd_pca']))
        cca_operator, d_pco = list(map(joblib.load, f_names))
        h_pco = joblib.load(os.path.join(base_dir, 'h_pca.pkl'))

        h_pred = np.load(os.path.join(base_dir, 'roots_whpa', f'{root}.npy'))

        # Inspect transformation between physical and PC space
        dnc0 = d_pco.ncomp
        hnc0 = h_pco.ncomp

        # Cut desired number of PC components
        d_pc_training, d_pc_prediction = d_pco.pca_refresh(dnc0)
        h_pco.pca_test_transformation(h_pred, test_root=[root])
        h_pc_training, h_pc_prediction = h_pco.pca_refresh(hnc0)

        # CCA plots
        d_cca_training, h_cca_training = cca_operator.transform(d_pc_training, h_pc_training)
        d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T

        cca_coefficient = np.corrcoef(d_cca_training, h_cca_training,).diagonal(offset=cca_operator.n_components)

        # cca_plot(cca_operator, d_cca_training, h_cca_training, d_pc_prediction, h_pc_prediction,
        #          sdir=os.path.join(os.path.dirname(res_dir), 'cca'))

        sns.lineplot(data=cca_coefficient)
        plt.grid(alpha=.2, linewidth=.5)
        plt.title('Decrease of CCA correlation coefficient with component number')
        plt.ylabel('Correlation coefficient')
        plt.xlabel('Component number')
        plt.savefig(os.path.join(os.path.dirname(res_dir), 'cca', 'coefs.png'), dpi=300, transparent=True)
        plt.show()


def plot_whpa(root=None):
    """
    Loads target pickle and plots all training WHPA
    :param root:
    :return:
    """

    if isinstance(root, (list, tuple)):
        if len(root) > 1:
            print('Input error')
            return
        else:
            root = root[0]

    base_dir = os.path.join(MySetup.Directories.forecasts_dir, 'base')
    x_lim, y_lim, grf = MySetup.Focus.x_range, MySetup.Focus.y_range, MySetup.Focus.cell_dim
    mplot = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf)

    fobj = os.path.join(MySetup.Directories.forecasts_dir, 'base', 'h_pca.pkl')
    h = joblib.load(fobj)
    h_training = h.training_physical.reshape(h.shape)

    mplot.whp(h_training)

    if root is not None:
        h_pred = np.load(os.path.join(base_dir, 'roots_whpa', f'{root}.npy'))
        mplot.whp(h=h_pred, colors='red', lw=1, alpha=1,
                  fig_file=os.path.join(MySetup.Directories.forecasts_dir, 'base', 'whpa_training.png'))


def plot_pc_ba(root, data=False, target=False):
    """

    :param root:
    :param data:
    :param target:
    :return:
    """

    if isinstance(root, (list, tuple)):
        if len(root) > 1:
            print('Input error')
            return
        else:
            root = root[0]

    x_lim, y_lim, grf = MySetup.Focus.x_range, MySetup.Focus.y_range, MySetup.Focus.cell_dim
    mplot = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf)

    base_dir = os.path.join(MySetup.Directories.forecasts_dir, 'base')
    if target:
        fobj = os.path.join(base_dir, 'h_pca.pkl')
        h_pco = joblib.load(fobj)
        hnc0 = h_pco.ncomp
        # mplot.h_pca_inverse_plot(h_pco, hnc0, training=True, fig_dir=os.path.join(base_dir, 'control'))

        h_pred = np.load(os.path.join(base_dir, 'roots_whpa', f'{root}.npy'))
        # Cut desired number of PC components
        h_pco.pca_test_transformation(h_pred, test_root=[root])
        h_pco.pca_refresh(hnc0)
        mplot.h_pca_inverse_plot(h_pco, hnc0, training=False, fig_dir=os.path.join(base_dir, 'control'))

    # d
    if data:
        subdir = os.path.join(MySetup.Directories.forecasts_dir, root)
        listme = os.listdir(subdir)
        folders = list(filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))

        for f in folders:
            res_dir = os.path.join(subdir, f, 'obj')
            # Load objects
            d_pco = joblib.load(os.path.join(res_dir, 'd_pca.pkl'))
            dnc0 = d_pco.ncomp
            d_pco.pca_refresh(dnc0)
            setattr(d_pco, 'test_root', [root])
            mplot.d_pca_inverse_plot(d_pco, dnc0, training=True,
                                     fig_dir=os.path.join(os.path.dirname(res_dir), 'pca'))
            mplot.d_pca_inverse_plot(d_pco, dnc0, training=False,
                                     fig_dir=os.path.join(os.path.dirname(res_dir), 'pca'))


def main(samples):
    for sample in samples:
        plot_pc_ba(sample, data=True, target=True)
        empty_figs(sample)
        plot_whpa(sample)
        cca_vision(sample, folders=None)
        pca_vision(sample, d=True, h=True, exvar=True, scores=True, folders=None)


if __name__ == '__main__':
    # TODO: Adapt for several observations
    base_dir = os.path.join(MySetup.Directories.forecasts_dir, 'base')
    test_roots = datread(os.path.join(base_dir, 'test_roots.dat'))
    main(test_roots)


