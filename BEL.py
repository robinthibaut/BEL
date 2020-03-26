# Import
import os
from os.path import join as jp
import shutil
import uuid
import warnings
import joblib

import numpy as np

from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import PowerTransformer

import matplotlib.pyplot as plt

from MyToolbox import FileOps, DataOps, MeshOps, Plot, PCAOps
from Tarantola import posterior

plt.style.use('dark_background')

mp = Plot()
do = DataOps()
mo = MeshOps()


def bel(n_training, n_test):

    # Directories
    new_dir = str(uuid.uuid4())  # sub-directory for figures
    cwd = os.getcwd()
    res_dir = jp(cwd, 'results')

    fig_dir = jp(cwd, 'figures', new_dir)
    fig_data_dir = jp(fig_dir, 'Data')
    fig_pca_dir = jp(fig_dir, 'PCA')
    fig_cca_dir = jp(fig_dir, 'CCA')
    fig_pred_dir = jp(fig_dir, 'Predictions')

    [FileOps.dirmaker(f) for f in [fig_data_dir, fig_pca_dir, fig_cca_dir, fig_pred_dir]]

    # Load data
    n = n_training + n_test  # Total number of simulations to load.
    check = False  # Flag to check for simulations
    flag = False  # Flag to load both d and h or only d
    if flag:
        tc0, roots = FileOps.load_data(res_dir=res_dir, n=n, check=check, data_flag=flag)
    else:
        tc0, h, roots = FileOps.load_data(res_dir=res_dir, n=n, check=check)

    # Save file roots
    with open(jp(fig_dir, 'roots.dat'), 'w') as f:
        for r in roots:
            f.write(os.path.basename(r)+'\n')

    # Preprocess d in n time steps.
    tc = do.d_process(tc0=tc0, n_time_steps=100)
    n_wel = len(tc[0])  # Number of injecting wels

    # Plot d
    mp.curves(tc=tc, n_wel=n_wel, sdir=fig_data_dir)
    mp.curves_i(tc=tc, n_wel=n_wel, sdir=fig_data_dir)

    # Preprocess h - the signed distance array comes with 1m cell dimension, we average the value by averaging 5
    # cells in both directions.
    do.h_process(h, sc=5, wdir=jp(cwd, 'temp'))
    h_u = np.load(jp(cwd, 'temp', 'h_u.npy'))
    h = h_u.copy()

    # Plot all WHPP
    mp.whp(h, fig_file=jp(fig_data_dir, 'all_whpa.png'), show=True)

    # %%  PCA

    # Choose size of training and prediction set
    n_sim = len(h)  # Number of simulations
    n_obs = n_test  # Number of 'observations' on which the predictions will be made.
    n_training = n_sim - n_obs  # number of synthetic data that will be used for constructing our prediction model

    if n_training != n_sim - n_obs:
        warnings.warn("The size of training set doesn't correspond with user input")

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
    mp.explained_variance(d_pco.operator, n_comp=50, fig_file=jp(fig_pca_dir, 'd_exvar.png'), show=True)
    mp.explained_variance(h_pco.operator, n_comp=50, fig_file=jp(fig_pca_dir, 'h_exvar.png'), show=True)

    # Scores plots
    mp.pca_scores(d_pc_training, d_pc_prediction, n_comp=20, fig_file=jp(fig_pca_dir, 'd_scores.png'), show=True)
    mp.pca_scores(h_pc_training, h_pc_prediction, n_comp=20, fig_file=jp(fig_pca_dir, 'h_scores.png'), show=True)

    # Choose number of PCA components to keep.
    # Compares true value with inverse transformation from PCA
    ndo = 45  # Number of components for breakthrough curves
    nho = 30  # Number of components for signed distance

    def pca_inverse_compare():
        n_compare = np.random.randint(n_training)  # Sample number to perform inverse transform comparison
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
        n_comp_cca = min(n_d_pc_comp, n_h_pc_comp)  # Number of CCA components is chosen as the min number of PC
        # components between d and h.
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
    mp.cca(cca, d_cca_training, h_cca_training, d_pc_prediction, h_pc_prediction, sdir=fig_cca_dir)

    # Pick an observation
    for sample_n in range(n_obs):
        d_pc_obs = d_pc_prediction[sample_n]
        h_pc_obs = h_pc_prediction[sample_n]
        h_true = h[n_training:][sample_n]  # True prediction

        # Project observed data into canonical space.
        d_cca_prediction, _ = cca.transform(d_pc_obs.reshape(1, -1), h_pc_obs.reshape(1, -1))
        d_cca_prediction = d_cca_prediction.T

        # Ensure Gaussian distribution in h_cca
        # Each vector for each cca components will be transformed one-by-one by a different operator, stored in yj.
        yj = [PowerTransformer(method='yeo-johnson', standardize=True) for c in range(h_cca_training.shape[0])]
        # Fit each PowerTransformer with each component
        [yj[i].fit(h_cca_training[i].reshape(-1, 1)) for i in range(len(yj))]
        # Transform the original distribution.
        h_cca_training_gaussian \
            = np.concatenate([yj[i].transform(h_cca_training[i].reshape(-1, 1)) for i in range(len(yj))], axis=1).T

        # Estimate the posterior mean and covariance (Tarantola)
        h_mean_posterior, h_posterior_covariance = posterior(h_cca_training_gaussian,
                                                             d_cca_training,
                                                             d_pc_training,
                                                             d_rotations,
                                                             d_cca_prediction)

        # %% Sample the posterior

        n_posts = 500  # Number of estimates sampled from the distribution.
        # Draw n_posts random samples from the multivariate normal distribution :
        h_posts_gaussian = np.random.multivariate_normal(mean=h_mean_posterior.T[0],
                                                         cov=h_posterior_covariance,
                                                         size=n_posts).T

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
        ff = jp(fig_pred_dir, '{}_{}_{}.png'.format(sample_n, add_comp, n_comp_cca))
        mp.whp_prediction(forecasts=forecast_posterior,
                          h_true=h_true,
                          h_pred=h_pred,
                          fig_file=ff)

        def save_forecasts():
            true_file = jp(cwd, 'temp', 'forecasts', '{}_true.npy'.format(sample_n))
            np.save(true_file, h_true)
            forecast_file = jp(cwd, 'temp', 'forecasts', '{}_forecasts.npy'.format(sample_n))
            np.save(forecast_file, forecast_posterior)

    shutil.copy(__file__, jp(fig_dir, 'copied_script.py'))


if __name__ == "__main__":
    # # Set numpy seed
    # seed = np.random.randint(0, 10e8)
    # np.random.seed(seed)
    bel(n_training=300, n_test=10)
