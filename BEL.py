# Import
import os
from os.path import join as jp

import joblib

import numpy as np
from numpy.matlib import repmat

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer

import matplotlib.pyplot as plt
plt.style.use('dark_background')

from MyToolbox import FileOps, DataOps, MeshOps, Plot
from Tarantola import posterior

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

tc0, h, hk = FileOps.load_data(res_dir=res_dir, n=0)

# Preprocess d
tc = do.d_process(tc0=tc0)

n_wel = len(tc[0])  # Number of injecting wels

# Plot d
# mp.curves(tc=tc, n_wel=n_wel, sdir=jp(cwd, 'figures', 'Data'))
# mp.curves_i(tc=tc, n_wel=n_wel, show=True)
# mp.curves_i(tc=tc, n_wel=n_wel, sdir=jp(cwd, 'figures', 'Data'))

# Preprocess h
# Let's first try to divide it using cells of side length = 5m
# Geometry
# xlim, ylim = 1500, 1000
# grf = 1  # Cell dimension (1m)
# nrow, ncol = ylim // grf, xlim // grf
# sc = 5
# un, uc = int(nrow / sc), int(ncol / sc)

# h_u = mo.h_sub(h, un, uc, sc)
# np.save(jp(cwd, 'temp', 'h_u'), h_u)  # Load transformed SD matrix
h_u = np.load(jp(cwd, 'temp', 'h_u.npy'))
h0 = h.copy()
h = h_u.copy()

# Plot all WHPP
# mp.whp(h, sc, jp(cwd, 'grid'), show=True)

# %%  PCA

n_sim = len(h)  # Number of simulations

n_training = int(n_sim * .98)  # number of synthetic data that will be used for constructing our prediction model
n_obs = n_sim - n_training

# PCA on transport curves
# Flattened, normalized, breakthrough curves
d_original = np.array([item for sublist in tc for item in sublist]).reshape(n_sim, -1)
d_training = d_original[:n_training]
d_prediction = d_original[n_training:]

d_pca_operator = PCA()  # The PCA is performed on all data (synthetic + 'observed')
d_pca_operator.fit(d_training)  # Principal components
joblib.dump(d_pca_operator, jp(cwd, 'temp', 'd_pca_operator.pkl'))  # Save the fitted PCA operator
d_pca_operator = joblib.load(jp(cwd, 'temp', 'd_pca_operator.pkl'))

d_pc_training = d_pca_operator.transform(d_training)  # Principal components
d_pc_prediction = d_pca_operator.transform(d_prediction)

#  PCA on signed distance
h_original = h.reshape(n_sim, -1)  # Not normalized
h_training = h_original[:n_training]
h_prediction = h_original[n_training:]

h_pca_operator = PCA()  # Try: Explain everything, keep all scores
h_pca_operator.fit(h_training)  # Principal components
joblib.dump(h_pca_operator, jp(cwd, 'temp', 'h_pca_operator.pkl'))  # Save the fitted PCA operator
h_pca_operator = joblib.load(jp(cwd, 'temp', 'h_pca_operator.pkl'))

h_pc_training = h_pca_operator.transform(h_training)  # Principal components
h_pc_prediction = h_pca_operator.transform(h_prediction)  # Selects curves until desired sample number

# Explained variance plots
# jp(cwd, 'figures', 'PCA', 'd_exvar.png')
# mp.explained_variance(d_pca_operator, n_comp=95, fig_file=jp(cwd, 'figures', 'PCA', 'd_exvar.png'), show=True)
# mp.explained_variance(d_pca_operator, n_comp=95, show=True)

# jp(cwd, 'figures', 'PCA', 'h_exvar.png')
# mp.explained_variance(h_pca_operator, n_comp=20, fig_file=jp(cwd, 'figures', 'PCA', 'h_exvar.png'), show=True)
# mp.explained_variance(h_pca_operator, n_comp=20, show=True)

# Scores plots

# jp(cwd, 'figures', 'PCA', 'd_scores.png')
# mp.pca_scores(d_pc_training, d_pc_prediction, n_comp=20, fig_file=jp(cwd, 'figures', 'PCA', 'd_scores.png'), show=True)

# jp(cwd, 'figures', 'PCA', 'h_scores.png')
# mp.pca_scores(h_pc_training, h_pc_prediction, n_comp=20, fig_file=jp(cwd, 'figures', 'PCA', 'h_scores.png'), show=True)

n_d_pc_comp = 500
n_h_pc_comp = 300

d_pc_training0 = d_pc_training.copy()
h_pc_training0 = h_pc_training.copy()
# d_pc_training, h_pc_training = d_pc_training0.copy(), h_pc_training0.copy()

d_pc_training = d_pc_training[:, :n_d_pc_comp]
h_pc_training = h_pc_training[:, :n_h_pc_comp]

d_pc_prediction0 = d_pc_prediction.copy()
h_pc_prediction0 = h_pc_prediction.copy()
# d_pc_prediction, h_pc_prediction = d_pc_prediction0.copy(), h_pc_prediction0.copy()

d_pc_prediction = d_pc_prediction[:, :n_d_pc_comp]
h_pc_prediction = h_pc_prediction[:, :n_h_pc_comp]

# %% CCA

n_comp_cca = 100
cca = CCA(n_components=n_comp_cca, scale=True, max_iter=int(500*2))  # By default, it scales the data
cca.fit(d_pc_training, h_pc_training)
joblib.dump(cca, jp(cwd, 'temp', 'cca.pkl'))  # Save the fitted CCA operator
cca = joblib.load(jp(cwd, 'temp', 'cca.pkl'))

# Returns x_scores, y_scores after fitting inputs.
d_cca_training, h_cca_training = cca.transform(d_pc_training, h_pc_training)
d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T

# Get the rotation matrices
d_rotations, h_rotations = cca.x_rotations_, cca.y_rotations_

# Correlation coefficients
# mp.cca(cca, d_cca_training, h_cca_training, d_pc_prediction, h_pc_prediction, sdir=jp(cwd, 'figures', 'CCA'))

# Pick an observation
for sample_n in range(1):
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
    # The value of h_pca_reverse are the score of PCA in the forecast space
    # To reverse data in the original space, perform the matrix multiplication between the data in the CCA space
    # with the y_loadings matrix. Because CCA scales the input, we must multiply the output by the y_std deviation
    # and add the y_mean
    # h_pca_reverse = np.matmul(h_posts.T, np.linalg.pinv(h_rotations))*cca.y_std_ + cca.y_mean_
    h_pca_reverse = np.matmul(h_posts.T, cca.y_loadings_.T) * cca.y_std_ + cca.y_mean_
    # h_pca_reverse_test = cca.inverse_transform(Y=h_posts.T)
    # np.array_equal(h_pca_reverse, h_pca_reverse_test)

    def pc_random():
        r_rows = np.random.choice(h_pc_training0.shape[0], n_posts)
        score_selection = h_pc_training0[r_rows, n_h_pc_comp:]
        test = [np.random.choice(score_selection[:, i]) for i in range(score_selection.shape[1])]
        return np.array(test)

    add_comp = 0

    if add_comp:
        rnpc = np.array([pc_random() for i in range(n_posts)])
        h_pca_reverse = np.array([np.concatenate((h_pca_reverse[i], rnpc[i])) for i in range(n_posts)])

    # Generate forecast in the initial dimension.
    # We use the initial decomposition to build the forecast.
    # Inverse transform the values with the PCA operator and rescale the h output.

    forecast_posterior = \
        (np.dot(h_pca_reverse, h_pca_operator.components_[:h_pca_reverse.shape[1], :]) +
         h_pca_operator.mean_).reshape((n_posts, h.shape[1], h.shape[2]))

    # forecast_posterior = \
    #      h_pca_operator.inverse_transform(h_pca_reverse).reshape((n_posts, h.shape[1], h.shape[2]))  # old

    # Predicting the SD based for a certain number of 'observations'
    h_pc_true_pred = cca.predict(d_pc_obs.reshape(1, -1))

    if add_comp:
        rnpc = pc_random()
        h_pc_true_pred = np.concatenate((h_pc_true_pred[0], rnpc)).reshape(1, -1)

    # Going back to the original SD dimension
    # h_pred = h_pca_operator.inverse_transform(h_pc_true_pred)  # old
    h_pred = np.dot(h_pc_true_pred, h_pca_operator.components_[:h_pc_true_pred.shape[1], :]) + h_pca_operator.mean_

    h_pred = h_pred.reshape(h.shape[1], h.shape[2])  # Reshape results

    # Plot results
    ff = jp(cwd, 'figures', 'Predictions', '{}prediction{}test.png'.format(sample_n, add_comp))
    mp.whp_prediction(forecast_posterior, h_true, h_pred, wdir=jp(cwd, 'grid'), fig_file=ff)
