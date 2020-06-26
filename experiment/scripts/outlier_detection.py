#  Copyright (c) 2020. Robin Thibaut, Ghent University

"""
The one-class SVM aims to fit a minimal volume
hypersphere around the samples in {d(1), d(2),…, d(L)}.
Any dobs that falls outside of this hypersphere is classified
as being inconsistent with the prior.

A one-class SVM using a Gaussian kernel is then trained
in functional space, and the decision boundary is
identified. If the observed data falls within this boundary,
then the prior is classified as being consistent with dobs.
"""


import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from experiment.toolbox import utils
from experiment.toolbox.filesio import load_res, folder_reset
from experiment.goggles.visualization import Plot, cca_plot, pca_scores, explained_variance
from experiment.base.inventory import MySetup

from sklearn import svm


def svm1(res_dir, d=True, h=False, folders=None):
    """ Loads PCA pickles and plot scores for all folders """
    subdir = os.path.join(MySetup.Directories.forecasts_dir, res_dir)
    if folders is None:
        listme = os.listdir(subdir)
        folders = list(filter(lambda du: os.path.isdir(os.path.join(subdir, du)), listme))
    else:
        if not isinstance(folders, (list, tuple)):
            folders = [folders]

    obj_ = []
    if d:

        for f in folders:
            dfig = os.path.join(subdir, f, 'pca')
            # For d only
            pcaf = os.path.join(subdir, f, 'obj', 'd_pca.pkl')
            d_pco = joblib.load(pcaf)
            obj_.append(d_pco)

    return obj_


sample = '6623dd4fb5014a978d59b9acb03946d2'
default = ['123456']

dpc = svm1(sample, folders=default)[0]

dataset = np.concatenate([dpc.training_pc[:, :2], dpc.predict_pc[:, :2]], axis=0)
dataset -= np.min(dataset)
dataset /= np.max(dataset)
dataset += 1


xx, yy = np.meshgrid(np.linspace(0.8, 2.2, 200),
                     np.linspace(0.8, 2.2, 200))

# 𝜈 is upper bounded by the fraction of outliers and lower bounded by the fraction of support vectors. Just consider
# that for default value 0.1 0.1 , atmost 10% of the training samples are allowed to be wrongly classified or
# can be considered as outliers by the decision boundary. And atleast 10% 10 % of your training samples will act as
# support vectors (points on the decision boundary).

# define outlier/anomaly detection methods to be compared
algorithm = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
algorithm.fit(dataset)
y_pred = algorithm.fit(dataset).predict(dataset)
Z = algorithm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='#ff7f00')
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu_r, alpha=.9)
cbar = plt.colorbar()
cbar.set_label('Distance to boundary')
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
b = plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange', alpha=.9)
plt.plot(dataset[:, 0], dataset[:, 1], 'wo', markersize=3, markeredgecolor='k')
plt.plot(dataset[-1, 0], dataset[-1, 1], 'ro', markersize=5, markeredgecolor='k')
plt.xlabel('First PC score')
plt.ylabel('Second PC score')

plt.show()
