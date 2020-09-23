#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import joblib
import numpy as np

from experiment.goggles.visualization import Plot
from experiment.base.inventory import MySetup as base

from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.metrics import mean_squared_error

fc = base.Focus()
x_lim, y_lim, grf = fc.x_range, fc.y_range, fc.cell_dim

mplot = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf, well_comb=None)

# def ng_boost(res_dir, folder=None):

res_dir, folder = '6623dd4fb5014a978d59b9acb03946d2', '123456'

subdir = os.path.join(base.Directories.forecasts_dir, res_dir)

#%% Load d
pcaf = os.path.join(subdir, folder, 'obj', 'd_pca.pkl')
d_pco = joblib.load(pcaf)
ndco = 50
d_pc_training = d_pco.training_pc[:, :ndco]
d_pc_test = d_pco.predict_pc.flatten()[:ndco].reshape(1, -1)

#%% Load h
hbase = os.path.join(base.Directories.forecasts_dir, 'base')
# Load h pickle
pcaf = os.path.join(hbase, 'h_pca.pkl')
h_pco = joblib.load(pcaf)
# Load npy whpa prediction
prediction = np.load(os.path.join(hbase, 'roots_whpa', f'{res_dir}.npy'))
# Transform and split
h_pco.pca_test_transformation(prediction, test_root=[res_dir])
nho = h_pco.ncomp
h_pc_training, h_pc_prediction = h_pco.pca_refresh(nho)

nhco = 37

my_pcs = []
# Try this NGboost
for i in range(nhco):
    ngb = NGBRegressor(Dist=Normal).fit(d_pc_training, h_pc_training[:, i])
    Y_dists = ngb.pred_dist(d_pc_test)
    my_pcs.append(Y_dists)

#%% Sample
n_samples = 500
random_samples = np.zeros((nhco, n_samples))

for i, d in enumerate(my_pcs):
    random_samples[i] = d.sample(n_samples)


random_samples = random_samples.reshape(n_samples, nhco)

# Generate forecast in the initial dimension and reshape.
forecast_posterior = h_pco.inverse_transform(random_samples[:, :15]).reshape((n_samples,
                                                                      h_pco.shape[1],
                                                                      h_pco.shape[2]))
mplot.whp(h=forecast_posterior, show_wells=True, show=True)

# test Mean Squared Error
# test_MSE = mean_squared_error(Y_preds, h_pc_prediction)
# print("Test MSE", test_MSE)
#
# # test Negative Log Likelihood
# test_NLL = -Y_dists.logpdf(h_pc_prediction.flatten()).mean()
# print("Test NLL", test_NLL)


# if __name__ == '__main__':
#     ng_boost('0cdb57a1b5dc4277b962d0bb289dbd48', '123456')
