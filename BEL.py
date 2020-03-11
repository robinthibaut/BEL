# Import
import os
from os.path import join as jp

import joblib

import numpy as np
from numpy.matlib import repmat

from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.spatial import distance_matrix

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
plt.style.use('dark_background')


from MyToolbox import FileOps, DataOps, MeshOps, Plot

mp = Plot()
do = DataOps()
mo = MeshOps()

# Directories
cwd = os.getcwd()
# This sets the main directory.
res_dir = jp(cwd, 'results')

# tpt, tc0, max_v, min_v, h, hk = AW.load_data()  # Function to load all saved data in binary format
# tpt = transport curves, tc = normalized and interpolated transport curves (1000 time steps on 200 days)
# sd = signed distance matrices, hk = hydraulic conductivity matrices

tc0, h, hk = FileOps.load_data(res_dir=res_dir)  # TODO: add pre-processing script, with Pipeline

# Preprocess d
tc = do.d_process(tc0=tc0)

n_wel = len(tc[0])  # Number of injecting wels

# Plot d
# mp.curves(tc=tc, n_wel=n_wel, sdir=jp(cwd, 'figures', 'Data'))
# mp.curves_i(tc=tc, n_wel=n_wel, sdir=jp(cwd, 'figures', 'Data'))

# Preprocess h
# Let's first try to divide it using cells of side length = 5m
# Geometry
xlim, ylim = 1500, 1000
grf = 1  # Cell dimension (1m)
nrow, ncol = ylim // grf, xlim // grf
sc = 5
un, uc = int(nrow / sc), int(ncol / sc)

# h_u = mo.h_sub(h, un, uc, sc)
# np.save(jp(cwd, 'temp', 'h_u'), h_u)  # Load transformed SD matrix
h_u = np.load(jp(cwd, 'temp', 'h_u.npy'))
h0 = h.copy()
h = h_u.copy()

# Plot all WHPP
mp.whp(h, sc, jp(cwd, 'grid'), show=True)

# %%  PCA

# TODO: Fix cutting method
# TODO: First apply PCA only on training set, explain 100% then cut until number of desired components
# TODO: Fix scores plot

n_sim = len(h)  # Number of simulations

n_training = int(n_sim * .99)  # number of synthetic data that will be used for constructing our prediction model
n_obs = n_sim - n_training

# PCA on transport curves
# Flattened, normalized, breakthrough curves
d_original = np.array([item for sublist in tc for item in sublist]).reshape(n_sim, -1)
d_training = d_original[:n_training]
d_prediction = d_original[n_training:]

# d_pca_operator = PCA()  # The PCA is performed on all data (synthetic + 'observed')
# d_pca_operator.fit(d_training)  # Principal components
# joblib.dump(d_pca_operator, jp(cwd, 'temp', 'd_pca_operator.pkl'))  # Save the fitted PCA operator
d_pca_operator = joblib.load(jp(cwd, 'temp', 'd_pca_operator.pkl'))

d_pc_training = d_pca_operator.transform(d_training)  # Principal components
d_pc_prediction = d_pca_operator.transform(d_prediction)

#  PCA on signed distance
h_original = h.reshape(n_sim, -1)  # Not normalized
h_training = h_original[:n_training]
h_prediction = h_original[n_training:]

# h_pca_operator = PCA()  # Try: Explain everything, keep all scores
# h_pca_operator.fit(h_training)  # Principal components
# joblib.dump(h_pca_operator, jp(cwd, 'temp', 'h_pca_operator.pkl'))  # Save the fitted PCA operator
h_pca_operator = joblib.load(jp(cwd, 'temp', 'h_pca_operator.pkl'))

h_pc_training = h_pca_operator.transform(h_training)  # Principal components
h_pc_prediction = h_pca_operator.transform(h_prediction)  # Selects curves until desired sample number

# Explained variance plots
# jp(cwd, 'figures', 'PCA', 'd_exvar.png')
# mp.explained_variance(d_pca_operator, n_comp=95, fig_file=jp(cwd, 'figures', 'PCA', 'd_exvar.png'), show=True)

# jp(cwd, 'figures', 'PCA', 'h_exvar.png')
# mp.explained_variance(h_pca_operator, n_comp=20, fig_file=jp(cwd, 'figures', 'PCA', 'h_exvar.png'), show=True)

# Scores plots

# jp(cwd, 'figures', 'PCA', 'd_scores.png')
# mp.pca_scores(d_pc_training, d_pc_prediction, n_comp=20, fig_file=jp(cwd, 'figures', 'PCA', 'd_scores.png'), show=True)

# jp(cwd, 'figures', 'PCA', 'h_scores.png')
# mp.pca_scores(h_pc_training, h_pc_prediction, n_comp=20, fig_file=jp(cwd, 'figures', 'PCA', 'h_scores.png'), show=True)

n_d_pc_comp = 95
n_h_pc_comp = 20

d_pc_training0 = d_pc_training.copy()
h_pc_training0 = h_pc_training.copy()
# d_pc_training, h_pc_training = d_pc_training0.copy(), h_pc_training0.copy()

d_pc_training = d_pc_training[:, :n_d_pc_comp]
h_pc_training = h_pc_training[:, :n_h_pc_comp]

d_pc_prediction0 = d_pc_prediction.copy()
h_pc_prediction0 = h_pc_prediction.copy()
# d_pc_training, h_pc_training = d_pc_training0.copy(), h_pc_training0.copy()

d_pc_prediction = d_pc_prediction[:, :n_d_pc_comp]
h_pc_prediction = h_pc_prediction[:, :n_h_pc_comp]

# %% CCA

# n_comp_cca = 20
# cca = CCA(n_components=n_comp_cca, scale=True, max_iter=int(500*1.5))  # By default, it scales the data
# cca.fit(d_pc_training, h_pc_training)
# joblib.dump(cca, jp(cwd, 'temp', 'cca.pkl'))  # Save the fitted CCA operator
cca = joblib.load(jp(cwd, 'temp', 'cca.pkl'))

# Returns x_scores, y_scores after fitting inputs.
d_cca_training, h_cca_training = cca.transform(d_pc_training, h_pc_training)
d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T

# Get the rotation matrices
d_rotations, h_rotations = cca.x_rotations_, cca.y_rotations_

# Correlation coefficients
mp.cca(cca, d_cca_training, h_cca_training, d_pc_prediction, h_pc_prediction, sdir=jp(cwd, 'figures', 'CCA'))

# Pick an observation
for sample_n in range(n_obs):
    d_pc_obs = d_pc_prediction[sample_n]  # The rests is considered as field data
    h_pc_obs = h_pc_prediction[sample_n]  # The rests is considered as field data
    h_true = h[n_training:][sample_n]  # True prediction
    hk_true = hk[n_training:][sample_n]  # Hydraulic conductivity field for the observed data

    # Project observed data into canonical space.
    d_cca_prediction, h_cca_prediction = cca.transform(d_pc_obs.reshape(1, -1), h_pc_obs.reshape(1, -1))
    d_cca_prediction, h_cca_prediction = d_cca_prediction.T, h_cca_prediction.T

    # Ensure Gaussian distribution in h_cca
    # Each vector for each cca components will be transformed one-by-one by a different operator, stored in yj.
    yj = [PowerTransformer(method='yeo-johnson', standardize=True) for c in range(h_cca_training.shape[0])]
    # Fit each PowerTransformer with each component
    [yj[i].fit(h_cca_training[i].reshape(-1, 1)) for i in range(len(yj))]
    # Transform the original distribution.
    h_cca_training_gaussian \
        = np.concatenate([yj[i].transform(h_cca_training[i].reshape(-1, 1)) for i in range(len(yj))], axis=1).T

    # Apply the transformation on the prediction as well.
    h_cca_prediction_gaussian \
        = np.concatenate([yj[i].transform(h_cca_prediction[i].reshape(-1, 1)) for i in range(len(yj))], axis=1).T

    # %% Predictions

    # Evaluate the covariance in h
    h_cov_operator = np.cov(h_cca_training_gaussian.T, rowvar=False)  # same

    # Evaluate the covariance in d (here we assume no data error, so C is identity times a given factor)
    x_dim = np.size(d_pc_training, axis=1)  # Number of PCA components for the curves
    noise = .01
    d_cov_operator = np.eye(x_dim) * noise  # I matrix.
    d_noise_covariance = d_rotations.T @ d_cov_operator @ d_rotations  # same

    # Linear modeling d to h
    g = np.linalg.lstsq(h_cca_training_gaussian.T, d_cca_training.T, rcond=None)[0].T  # Transpose to get same as Thomas
    g = np.where(np.abs(g) < 1e-12, 0, g)

    # Modeling error due to deviations from theory
    d_ls_predicted = g @ h_cca_training_gaussian  # same
    d_modeling_mean_error = np.mean(d_cca_training - d_ls_predicted, axis=1).reshape(-1, 1)  # same
    d_modeling_error = d_cca_training - d_ls_predicted - repmat(d_modeling_mean_error, 1,
                                                                np.size(d_cca_training, axis=1))  # same
    # Information about the covariance of the posterior distribution.
    d_modeling_covariance = (d_modeling_error @ d_modeling_error.T) / n_training  # same

    # Computation of the posterior mean
    h_mean = np.row_stack(np.mean(h_cca_training_gaussian, axis=1))  # same
    h_mean = np.where(np.abs(h_mean) < 1e-12, 0, h_mean)  # My mean is 0, as expected.

    h_mean_posterior = h_mean \
                       + h_cov_operator @ g.T \
                       @ np.linalg.pinv(g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance) \
                       @ (d_cca_prediction.reshape(-1, 1) + d_modeling_mean_error - g @ h_mean)  # same

    # h posterior covariance
    h_posterior_covariance = h_cov_operator \
                             - (h_cov_operator @ g.T) \
                             @ np.linalg.pinv(g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance) \
                             @ g @ h_cov_operator

    h_posterior_covariance = (h_posterior_covariance + h_posterior_covariance.T) / 2  # same

    # %% Sample the posterior

    n_posts = 500  # Number of estimates sampled from the distribution.
    # Draw n_posts random samples from the multivariate normal distribution :
    h_posts_gaussian = np.random.multivariate_normal(mean=h_mean_posterior.T[0],
                                                     cov=h_posterior_covariance,
                                                     size=n_posts).T  # seems OK

    # This h_posts gaussian need to be inverse-transformed to the original distribution.
    # We get the CCA scores.
    h_posts \
        = np.concatenate([yj[i].inverse_transform(h_posts_gaussian[i].reshape(-1, 1)) for i in range(len(yj))],
                         axis=1).T

    # Calculate the values of hf, i.e. reverse the canonical correlation, it always works if dimf > dimh
    # The value of h_pca_reverse are the score of PCA in the forecast space
    # To reverse data in the original space, perform the matrix multiplication between the data in the CCA space
    # with the y_loadings matrix. Because CCA scales the input, we must multiply the output by the y_std deviation
    # and add the y_mean
    # h_pca_reverse = np.matmul(h_posts.T, np.linalg.pinv(h_rotations))*cca.y_std_ + cca.y_mean_
    h_pca_reverse = np.matmul(h_posts.T, cca.y_loadings_.T) * cca.y_std_ + cca.y_mean_
    # h_pca_reverse_test = cca.inverse_transform(Y=h_posts.T)
    # np.array_equal(h_pca_reverse, h_pca_reverse_test)

    # Generate forecast in the initial dimension.
    # We use the initial decomposition to build the forecast.
    # Inverse transform the values with the PCA operator and rescale the h output.

    forecast_posterior = \
        (np.dot(h_pca_reverse, h_pca_operator.components_[:n_h_pc_comp, :]) +
         h_pca_operator.mean_).reshape((n_posts, h.shape[1], h.shape[2]))

    # forecast_posterior \
    #     = h_pca_operator.inverse_transform(h_pca_reverse).reshape((n_posts, h.shape[1], h.shape[2]))

    # Predicting the SD based for a certain number of 'observations'

    h_pc_true_pred = cca.predict(d_pc_obs.reshape(1, -1))

    # Going back to the original SD dimension
    # h_pred = h_pca_operator.inverse_transform(h_pc_true_pred)
    # TODO: Add here the rest of the PCA components
    h_pred = np.dot(h_pc_true_pred, h_pca_operator.components_[:n_h_pc_comp, :]) + h_pca_operator.mean_
    h_pred = h_pred.reshape(h.shape[1], h.shape[2])  # Reshape results

    # Plot results
    ff = jp(cwd, 'figures', 'Predictions', '{}prediction.png'.format(sample_n))
    mp.whp_prediction(sc, forecast_posterior, h_true, h_pred, wdir=jp(cwd, 'grid'), fig_file=ff)

    # grf = 5
    # X, Y = np.meshgrid(np.linspace(0, xlim, int(xlim / grf)), np.linspace(0, ylim, int(ylim / grf)))
    # Z = np.copy(h_true)
    # plt.subplots()
    # plt.grid(color='c', linestyle='-', linewidth=.5, alpha=0.4)
    # # Plot n sampled forecasts
    # for z in forecast_posterior:
    #     plt.contour(X, Y, z, [0], colors='white', alpha=0.1)
    # # Plot true h
    # plt.contour(X, Y, Z, [0], colors='red', linewidths=1, alpha=.9)
    # # Plot true h predicted
    # plt.contour(X, Y, h_true_pred[0], [0], colors='cyan', linewidths=1, alpha=.9)
    # # Plot hk
    # # plt.imshow(hk_true[0], origin='lower', extent=(0, xlim, 0, ylim),
    # #            norm=LogNorm(vmin=np.min(hk_true[0]), vmax=np.max(hk_true[0])),
    # #            cmap='coolwarm', alpha=0.7)
    # # plt.colorbar()
    # # Plot wells
    # pwl = np.load((jp(cwd, 'grid', 'pw.npy')), allow_pickle=True)[:, :2]
    # plt.plot(pwl[0][0], pwl[0][1], 'wo', label='pw')
    # iwl = np.load((jp(cwd, 'grid', 'iw.npy')), allow_pickle=True)[:, :2]
    # for i in range(len(iwl)):
    #     plt.plot(iwl[i][0], iwl[i][1],
    #              'o', markersize=4, markeredgecolor='k', markeredgewidth=.5,
    #              label='iw{}'.format(i))
    # plt.tick_params(labelsize=5)
    # plt.legend(fontsize=7, loc=2)
    # plt.xlim(750, 1200)
    # plt.ylim(300, 700)
    # plt.savefig(jp(cwd, 'figures', 'Predictions', '{}prediction.png'.format(sample_n)), bbox_inches='tight', dpi=300)
    # plt.show()


# palette = ['tab:{}'.format(c) for c in
#            ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'olive', 'cyan']]
# np.random.shuffle(palette)
# h_obs = h[n_training:]  # True predictions
# grf = 5
# X, Y = np.meshgrid(np.linspace(0, xlim, int(xlim / grf)), np.linspace(0, ylim, int(ylim / grf)))
# Z = np.copy(h_obs)
# plt.subplots()
# plt.grid(color='w', linestyle='-', linewidth=1, alpha=0.2)
# for sd in h[:n_training]:
#     plt.contour(X, Y, sd, [0], colors='white', linewidths=.5, alpha=.2)
# for z in range(n_obs):
#     plt.contour(X, Y, h_obs[z], [0], colors=palette[z], linewidths=.8, alpha=1)
# # plt.legend()
# pwl = np.load((jp(cwd, 'grid', 'pw.npy')), allow_pickle=True)[:, :2]
# plt.plot(pwl[0][0], pwl[0][1], 'wo', label='pw')
# iwl = np.load((jp(cwd, 'grid', 'iw.npy')), allow_pickle=True)[:, :2]
# for i in range(len(iwl)):
#     plt.plot(iwl[i][0], iwl[i][1],
#              'o', markersize=5, markeredgecolor='k', markeredgewidth=.5,
#              label='iw{}'.format(i))
# plt.tick_params(labelsize=8)
# plt.legend(fontsize=7, loc=2)
# plt.xlim(750, 1200)
# plt.ylim(300, 700)
# plt.savefig(jp(cwd, 'figures', 'Predictions', 'observations.png'), bbox_inches='tight', dpi=300)
# plt.show()
