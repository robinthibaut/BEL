#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from experiment.toolbox import utils
from experiment.toolbox.filesio import load_res, folder_reset
from experiment.goggles.visualization import Plot, cca_plot, pca_scores, explained_variance
from experiment.base.inventory import MySetup


def dists(res_dir, folders=None):
    subdir = os.path.join(MySetup.Directories.forecasts_dir, res_dir)
    if folders is None:
        listme = os.listdir(subdir)
        folders = list(filter(lambda du: os.path.isdir(os.path.join(subdir, du)), listme))
    else:
        if not isinstance(folders, (list, tuple)):
            folders = [folders]

    obj_ = []

    for f in folders:
        # For d only
        pcnpy = os.path.join(subdir, f, 'uq', '500_target_pc.npy')
        h_pc = np.load(pcnpy)
        obj_.append(h_pc)

    return obj_


sample = '6623dd4fb5014a978d59b9acb03946d2'
default = ['123456']

dpc_post = dists(sample, folders=default)[0]
hp1 = dpc_post[:, 0]

# %%

hbase = os.path.join(MySetup.Directories.forecasts_dir, 'base')
# Load h pickle
pcaf = os.path.join(hbase, 'h_pca.pkl')
h_pco = joblib.load(pcaf)

h1 = h_pco.training_pc[:, 0]

# %%


