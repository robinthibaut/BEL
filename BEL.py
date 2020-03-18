# Import
import os
from os.path import join as jp

import joblib

import numpy as np

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer

import matplotlib.pyplot as plt

from MyToolbox import FileOps, DataOps, MeshOps, Plot, PCAOps
from Tarantola import posterior

plt.style.use('dark_background')

mp = Plot()
do = DataOps()
mo = MeshOps()

# Directories
cwd = os.getcwd()
res_dir = jp(cwd, 'results')

# Load data
flag = True
if flag:
    tc0 = FileOps.load_data(res_dir=res_dir, n=0, data_flag=True)
else:
    tc0, h = FileOps.load_data(res_dir=res_dir, n=0)

# Preprocess d in n time steps.
tc = do.d_process(tc0=tc0)
n_wel = len(tc[0])  # Number of injecting wels

# Plot d
mp.curves(tc=tc, n_wel=n_wel, sdir=jp(cwd, 'figures', 'Data'))
mp.curves_i(tc=tc, n_wel=n_wel, sdir=jp(cwd, 'figures', 'Data'))

# Preprocess h - the signed distance array comes with 1m cell dimension, we average the value by averaging 5 cells in
# both directions.
do.h_process(h, sc=5, wdir=jp(cwd, 'temp'))
h_u = np.load(jp(cwd, 'temp', 'h_u.npz'))
h = h_u.copy()

# Plot all WHPP
mp.whp(h, fig_file=jp(cwd, 'figures', 'Data', 'all_whpa.pdf'), show=True)

# %%  PCA

# Choose size of training and prediction set
n_sim = len(h)  # Number of simulations
n_training = int(n_sim * .99)  # number of synthetic data that will be used for constructing our prediction model
n_obs = n_sim - n_training  # Number of 'observations' on which the predictions will be made.

load = False  # Whether to load already dumped PCA operator

# PCA on transport curves
d_pco = PCAOps(name='d', raw_data=tc)
d_training, d_prediction = d_pco.pca_tp(n_training)  # Split into training and prediction
d_pc_training, d_pc_prediction = d_pco.pca_transformation(load=load)

# PCA on signed distance
h_pco = PCAOps(name='h', raw_data=h)
h_training, h_prediction = h_pco.pca_tp(n_training)
h_pc_training, h_pc_prediction = h_pco.pca_transformation(load=load)

# Explained variance plots
mp.explained_variance(d_pco.operator, n_comp=50, fig_file=jp(cwd, 'figures', 'PCA', 'd_exvar.png'), show=True)
mp.explained_variance(h_pco.operator, n_comp=50, fig_file=jp(cwd, 'figures', 'PCA', 'h_exvar.png'), show=True)

# Scores plots
mp.pca_scores(d_pc_training, d_pc_prediction, n_comp=20, fig_file=jp(cwd, 'figures', 'PCA', 'd_scores.png'), show=True)
mp.pca_scores(h_pc_training, h_pc_prediction, n_comp=20, fig_file=jp(cwd, 'figures', 'PCA', 'h_scores.png'), show=True)

# Choose number of PCA components to keep.
# Compares true value with inverse transformation from PCA
ndo = 70  # Number of components for breakthrough curves
nho = 35  # Number of components for signed distance
n_compare = 0  # Sample number to perform inverse tranform comparison
mp.d_pca_inverse_plot(d_training, n_compare, d_pco.operator, ndo)
mp.h_pca_inverse_plot(h_training, n_compare, h_pco.operator, nho)

# Displays the explained variance percentage given the number of components
print(d_pco.perc_pca_components(ndo))
print(h_pco.perc_pca_components(nho))

# Assign final n_comp
n_d_pc_comp = ndo
n_h_pc_comp = nho

# Cut desired number of PC components
d_pc_training, d_pc_prediction = d_pco.pca_refresh(n_d_pc_comp)
h_pc_training, h_pc_prediction = h_pco.pca_refresh(n_h_pc_comp)

# %% CCA

load_cca = False

if not load_cca:
    n_comp_cca = min(n_d_pc_comp, n_h_pc_comp)  # Number of CCA components is chosen as the min number of PC components
    # between d and h.
    cca = CCA(n_components=n_comp_cca, scale=True, max_iter=int(500*2))  # By default, it scales the data
    cca.fit(d_pc_training, h_pc_training)  # Fit
    joblib.dump(cca, jp(cwd, 'temp', 'cca.pkl'))  # Save the fitted CCA operator
else:
    cca = joblib.load(jp(cwd, 'temp', 'cca.pkl'))
    n_comp_cca = cca.n_components_

# Returns x_scores, y_scores after fitting inputs.
d_cca_training, h_cca_training = cca.transform(d_pc_training, h_pc_training)
d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T

# Get the rotation matrices
d_rotations, h_rotations = cca.x_rotations_, cca.y_rotations_

# Correlation coefficients plots
mp.cca(cca, d_cca_training, h_cca_training, d_pc_prediction, h_pc_prediction, sdir=jp(cwd, 'figures', 'CCA'))

# Pick an observation
for sample_n in range(n_obs):
    d_pc_obs = d_pc_prediction[sample_n]
    h_pc_obs = h_pc_prediction[sample_n]
    h_true = h[n_training:][sample_n]  # True prediction

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

    # Estimate the posterior mean and covariance (Tarantola)
    h_mean_posterior, h_posterior_covariance = posterior(h_cca_training_gaussian,
                                                         d_cca_training,
                                                         d_pc_training,
                                                         d_rotations,
                                                         d_cca_prediction)

    # %% Sample the posterior

    n_posts = 500  # Number of estimates sampled from the distribution.
    # np.random.seed(42*(sample_n+1))
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
    # The value of h_pca_reverse are the score of PCA in the forecast space.
    # To reverse data in the original space, perform the matrix multiplication between the data in the CCA space
    # with the y_loadings matrix. Because CCA scales the input, we must multiply the output by the y_std deviation
    # and add the y_mean.
    # h_pca_reverse = np.matmul(h_posts.T, np.linalg.pinv(h_rotations))*cca.y_std_ + cca.y_mean_
    h_pca_reverse = np.matmul(h_posts.T, cca.y_loadings_.T) * cca.y_std_ + cca.y_mean_

    add_comp = 0  # Whether to add or not the rest of PC components

    if add_comp:
        rnpc = np.array([h_pco.pc_random(n_posts) for i in range(n_posts)])
        h_pca_reverse = np.array([np.concatenate((h_pca_reverse[i], rnpc[i])) for i in range(n_posts)])

    # Generate forecast in the initial dimension and reshape.
    forecast_posterior = h_pco.inverse_transform(h_pca_reverse).reshape((n_posts, h.shape[1], h.shape[2]))

    # Predicting the SD based for a certain number of 'observations'
    h_pc_true_pred = cca.predict(d_pc_obs.reshape(1, -1))

    # Going back to the original SD dimension and reshape.
    h_pred = h_pco.inverse_transform(h_pc_true_pred).reshape(h.shape[1], h.shape[2])

    # Plot results
    # ff = jp(cwd, 'figures', 'Predictions', '{}_{}_{}.png'.format(sample_n, add_comp, n_comp_cca))
    # mp.whp_prediction(forecasts=forecast_posterior,
    #                   h_true=h_true,
    #                   h_pred=h_pred,
    #                   fig_file=ff)

    forecast_file = jp(cwd, 'temp', 'forecasts', '{}_forecasts.npz'.format(sample_n))
    np.savez_compressed(forecast_file, forecast_posterior)
