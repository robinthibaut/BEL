# %% Import
import os
from os.path import join as jp

import numpy as np
from numpy.matlib import repmat
from scipy.io import loadmat
from scipy.interpolate import interp1d
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

from MyToolbox import MyWorkshop

# Geometry of the signed distance data:
# 1500 m in x direction
# 1000 m in y direction
# Cell size (square) is 1m
cols = ['b', 'g', 'r', 'c', 'm']  # Color list

# %% Directories
cwd = os.getcwd()

# This sets the main directory.
res_dir = jp(cwd, 'results')

# for root, dirs, files in os.walk(res_dir, topdown=False):
#     for name in files:
#         print(root)

xlim = 1500
ylim = 1000
grf = 1  # Cell dimension (1m)

AW = MyWorkshop()
AW.res_dir = res_dir
AW.xlim = xlim
AW.ylim = ylim
AW.grf = grf

# tpt, tc0, max_v, min_v, h, hk = AW.load_data()  # Function to load all saved data in binary format
# tpt = transport curves, tc = normalized and interpolated transport curves (1000 time steps on 200 days)
# sd = signed distance matrices, hk = hydraulic conductivity matrices

tc0, h, hk = AW.load_data()  # TODO: add pre-processing script
f1d = []  # List of interpolating functions for each curve
for t in tc0:
    fs = [interp1d(c[:, 0], c[:, 1], fill_value='extrapolate') for c in t]
    f1d.append(fs)
f1d = np.array(f1d)
# Watch out as the two following variables are also defined in the load_data() function:
n_time_steps = 500  # Arbitrary number of time steps to create the final transport array
ls = np.linspace(0, 1.01080e+02, num=n_time_steps)  # From 0 to 200 days with 1000 steps
tc = []  # List of interpolating functions for each curve
for f in f1d:
    ts = [fi(ls) for fi in f]
    tc.append(ts)
tc = np.array(tc)

# import matplotlib.pyplot as plt
# for i in range(len(tc)):
#     for t in tc[i]:
#         plt.plot(t)
#     plt.show()

n_sim = len(h)  # Number of simulations
n_wel = len(tc[0])  # Number of injecting wels

# %% PCA

n_training = 280  # number of synthetic data that will be used for constructing our prediction model
n_obs = n_sim - n_training - 1
sample_n = 0

# PCA on transport curves
# Flattened, normalized, breakthrough curves
d_original = np.array([item for sublist in tc for item in sublist]).reshape(n_sim, n_time_steps * n_wel)

d_pca_operator = PCA(.999)  # The PCA is performed on all data (synthetic + 'observed')

d_pc_scores = d_pca_operator.fit_transform(d_original)  # Principal components

d_pc_training = d_pc_scores[:n_training]  # Selects curves until desired sample number

d_pc_prediction = d_pc_scores[n_training:][sample_n]  # The rests is considered as field data

#  PCA on signed distance
h_pca_operator = PCA(.97)  # TODO: Explain everything, keep all scores

h_original = h.reshape((n_sim, h.shape[1] * h.shape[2]))  # Not normalized

h_pc_scores = h_pca_operator.fit_transform(h_original)  # Principal components

h_pc_training = h_pc_scores[:n_training]  # Selects curves until desired sample number

h_pc_prediction = h_pc_scores[n_training:][sample_n]  # The rests is considered as field data

h_true = h[n_training:][sample_n]  # True prediction

hk_true = hk[n_training:][sample_n]  # Hydraulic conductivity field for the observed data

# Data from Thomas:
# A_t = loadmat(jp(md, 'misc', 'A_thomas.mat'))['A']
# B_t = loadmat(jp(md, 'misc', 'B_thomas.mat'))['B']
# U_t = loadmat(jp(md, 'misc', 'U_thomas.mat'))['Dc']
# V_t = loadmat(jp(md, 'misc', 'V_thomas.mat'))['Hc']
# df = loadmat(jp(md, 'misc', 'Df.mat'))['Df']
# hf = loadmat(jp(md, 'misc', 'Hf.mat'))['Hf']
# dobs_f = loadmat(jp(md, 'misc', 'dobs_f.mat'))['dobs_f']
# hc_gauss = loadmat(jp(md, 'misc', 'hc_gauss.mat'))['Hc_gauss']
# ##############################################################################################
# d_pc_training = df
# h_pc_training = hf
# d_pc_prediction = dobs_f
# n_training = len(d_pc_training)

# %% CCA

n_comp_cca = 11  # Increasing this parameter will produce 'over-fitting effects'
# To obtain same results as Thomas, I need to not scale the data.
# However, after some tweaks, the code now works with the default settings.
cca = CCA(n_components=n_comp_cca, scale=True)  # By default, it scales the data

# Returns x_scores, y_scores after fitting inputs.
d_cca_training, h_cca_training = cca.fit_transform(d_pc_training, h_pc_training)
d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T

##############################################################################################
# d_cca_training = U_t
# h_cca_training = V_t

# Get the rotation matrices
d_rotations, h_rotations = cca.x_rotations_, cca.y_rotations_
##############################################################################################
# d_rotations, h_rotations = A_t, B_t

# Correlation coefficients
cca_coefficient = np.corrcoef(d_cca_training, h_cca_training).diagonal(offset=cca.n_components)

# Project observed data into canonical space.
d_cca_prediction, h_cca_prediction = cca.transform(d_pc_prediction.reshape(1, -1), h_pc_prediction.reshape(1, -1))
d_cca_prediction, h_cca_prediction = d_cca_prediction.T, h_cca_prediction.T

##############################################################################################
# d_cca_prediction = ((dobs_f - np.mean(d_pc_training, axis=0)) @ A_t).T

# CCA plot first component
# for i in range(n_comp_cca):
#     comp_n = i
#     plt.plot(d_cca_training[comp_n], h_cca_training[comp_n], 'ro')
#     plt.plot(d_cca_prediction[comp_n], h_cca_prediction[comp_n], 'ko')
#     plt.title(round(cca_coefficient[i], 4))
#     # plt.savefig(jp(cwd, '{}cca{}.png'.format(sample_n, i)), bbox_inches='tight', dpi=100)
#     plt.show()
#     plt.close()
comp_n = 0
# Ensure Gaussian distribution in h_cca ?
# np.random.RandomState(seed=7777777)
# Each vector for each cca components will be transformed one-by-one by a different operator, stored in yj.
yj = [PowerTransformer(method='yeo-johnson', standardize=True) for c in range(h_cca_training.shape[0])]
# Fit each PowerTransformer with each component
[yj[i].fit(h_cca_training[i].reshape(-1, 1)) for i in range(len(yj))]
# Transform the original distribution.
h_cca_training_gaussian \
    = np.concatenate([yj[i].transform(h_cca_training[i].reshape(-1, 1)) for i in range(len(yj))], axis=1).T

# CCA plot first component
# plt.hist(h_cca_training[comp_n], alpha=.5)
# plt.hist(h_cca_training_gaussian[comp_n], alpha=.5)
# plt.show()

# Apply the transformation on the prediction as well.
h_cca_prediction_gaussian \
    = np.concatenate([yj[i].transform(h_cca_prediction[i].reshape(-1, 1)) for i in range(len(yj))], axis=1).T

# CCA plot first component
# plt.plot(d_cca_training[comp_n], h_cca_training_gaussian[comp_n], 'ro')
# plt.plot(d_cca_prediction[comp_n], h_cca_prediction_gaussian[comp_n], 'ko')
# plt.show()

# %% Predictions
##############################################################################################
# h_cca_training_gaussian = hc_gauss  # Thomas data
# comp_n = 0
# plt.plot(d_cca_training[comp_n], h_cca_training_gaussian[comp_n], 'ro')
# plt.plot(d_cca_training[comp_n], h_cca_training[comp_n], 'ko')
# plt.show()
# for sample_n in range(2):

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

n_posts = 300  # Number of estimates sampled from the distribution.
# np.random.seed(88888888)
# Draw n_posts random samples from the multivariate normal distribution :
h_posts_gaussian = np.random.multivariate_normal(mean=h_mean_posterior.T[0],
                                                 cov=h_posterior_covariance,
                                                 size=n_posts).T  # seems OK

# This h_posts gaussian need to be inverse-transformed to the original distribution.
# We get the CCA scores.
h_posts \
    = np.concatenate([yj[i].inverse_transform(h_posts_gaussian[i].reshape(-1, 1)) for i in range(len(yj))], axis=1).T

# Calculate the values of hf, i.e. reverse the canonical correlation, it always works if dimf > dimh
# The value of h_pca_reverse are the score of PCA in the forecast space
# h_pca_reverse = np.matmul(h_posts.T, np.linalg.pinv(h_rotations))*cca.y_std_ + cca.y_mean_

# To reverse data in the original space, perform the matrix multiplication between the data in the CCA space
# with the y_loadings matrix. Because CCA scales the input, we must multiply the output by the y_std deviation
# and add the y_mean
h_pca_reverse = np.matmul(h_posts.T, cca.y_loadings_.T) * cca.y_std_ + cca.y_mean_

# h_pca_reverse_test = cca.inverse_transform(Y=h_posts.T)
# np.array_equal(h_pca_reverse, h_pca_reverse_test)
# Generate forecast in the initial dimension.
# We use the initial decomposition to build the forecast.
# Inverse transform the values with the PCA operator and rescale the h output.

forecast_posterior \
    = h_pca_operator.inverse_transform(h_pca_reverse).reshape((n_posts, h.shape[1], h.shape[2]))

# Predicting the SD based for a certain number of 'observations'
h_pc_true_pred = cca.predict(d_pc_prediction.reshape(1, -1))

# Going back to the original SD dimension
h_true_pred = h_pca_operator.inverse_transform(h_pc_true_pred)
h_true_pred = h_true_pred.reshape((len(h_true_pred), h.shape[1], h.shape[2]))  # Reshape results

# Plot results

X, Y = np.meshgrid(np.linspace(0, xlim, int(xlim / grf)), np.linspace(0, ylim, int(ylim / grf)))
Z = np.copy(h_true)
plt.subplots()
# Plot n sampled forecasts
for z in forecast_posterior:
    plt.contour(X, Y, z, [0], colors='blue', alpha=0.2)
# Plot true h
plt.contour(X, Y, Z, [0], colors='red')
# Plot true h predicted
plt.contour(X, Y, h_true_pred[0], [0], colors='yellow')
# Plot hk
plt.imshow(hk_true[0], origin='lower', extent=(0, 1500, 0, 1000),
           norm=LogNorm(vmin=np.min(hk_true[0]), vmax=np.max(hk_true[0])),
           cmap='coolwarm', alpha=0.7)
plt.colorbar()
# Plot wells
pwl = np.load((jp(cwd, 'grid', 'pw.npy')), allow_pickle=True)[:, :2]
plt.plot(pwl[0][0], pwl[0][1], 'ko', label='pw')
iwl = np.load((jp(cwd, 'grid', 'iw.npy')), allow_pickle=True)[:, :2]
for i in range(len(iwl)):
    plt.plot(iwl[i][0], iwl[i][1], 'o', label='iw{}'.format(i))
plt.legend()
plt.xlim(700, 1200)
plt.ylim(200, 800)
plt.savefig(jp(cwd, '{}prediction.png'.format(sample_n)), bbox_inches='tight', dpi=300)
plt.show()
