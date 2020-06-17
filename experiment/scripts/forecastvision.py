#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import joblib
import numpy as np
from experiment.toolbox.filesio import load_res
from experiment.goggles.visualization import Plot
from experiment.base.inventory import Directories, Focus

if __name__ == '__main__':
    x_lim, y_lim, grf = Focus.x_range, Focus.y_range, Focus.cell_dim
    mplot = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf)
    
    fobj = os.path.join(Directories.forecasts_dir, 'base', 'h_pca.pkl')
    h = joblib.load(fobj)
    h_training = h.training_physical.reshape(h.shape)

    mplot.whp(h_training, show=True,
              fig_file=os.path.join(Directories.forecasts_dir, 'base', 'whpa_training.png'))
