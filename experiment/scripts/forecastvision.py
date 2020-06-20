#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import joblib
import numpy as np
from experiment.toolbox import utils
from experiment.toolbox.filesio import load_res
from experiment.goggles.visualization import Plot, cca_plot
from experiment.base.inventory import MySetup


def cca_vision(res_dir):
    base_dir = os.path.join(MySetup.Directories.forecasts_dir, 'base')
    # Load objects
    f_names = list(map(lambda fn: os.path.join(res_dir, fn + '.pkl'), ['cca', 'd_pca']))
    cca_operator, d_pco = list(map(joblib.load, f_names))
    h_pco = joblib.load(os.path.join(base_dir, 'h_pca.pkl'))

    h_pred = np.load(os.path.join(base_dir, 'roots_whpa', '6623dd4fb5014a978d59b9acb03946d2.npy'))

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


if __name__ == '__main__':
    # x_lim, y_lim, grf = Focus.x_range, Focus.y_range, Focus.cell_dim
    # mplot = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf)
    #
    # fobj = os.path.join(Directories.forecasts_dir, 'base', 'h_pca.pkl')
    # h = joblib.load(fobj)
    # h_training = h.training_physical.reshape(h.shape)
    #
    # mplot.whp(h_training, show=True,
    #           fig_file=os.path.join(Directories.forecasts_dir, 'base', 'whpa_training.png'))

    comb = MySetup.Wels.combination  # Get default combination (all)

    subdir = os.path.join(MySetup.Directories.forecasts_dir, '6623dd4fb5014a978d59b9acb03946d2')
    listme = os.listdir(subdir)
    folders = list(filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))

    for f in folders:
        cca_vision(os.path.join(MySetup.Directories.forecasts_dir, '6623dd4fb5014a978d59b9acb03946d2', f, 'obj'))
