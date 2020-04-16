#  Copyright (c) 2020. Robin Thibaut, Ghent University

import numpy as np
from numpy.matlib import repmat

from bel.processing.data_ops import TargetOps


class PosteriorOps:

    def __init__(self):
        self.posterior_mean = 0
        self.posterior_covariance = 0
        self.ops = TargetOps()

    def posterior(self,
                  h_cca_training_gaussian,
                  d_cca_training,
                  d_pc_training,
                  d_rotations,
                  d_cca_prediction):
        """
        Estimating posterior uncertainties.

        Parameters
        ----------
        :param h_cca_training_gaussian:
               Canonical Variate of the training target, Gaussian-distributed
        :param d_cca_training:
               Canonical Variate of the training data
        :param d_pc_training:
               Principal Components of the training data
        :param d_rotations:
               CCA rotations of the training data
        :param d_cca_prediction:
               Canonical Variate of the observation

        Returns
        -------
        :return: h_mean_posterior, h_posterior_covariance

        Raises
        ------
        ValueError
            An exception is thrown if the shape of input arrays are not consistent.

        References
        ----------
        .. [1] A. Tarantola. Inverse Problem Theory and Methods for Model Parameter Estimation.
               SIAM, 2005. Pages: 70-71

        """

        # TODO: add dimension check
        # Size of the set
        n_training = d_cca_training.shape[1]

        # Evaluate the covariance in h
        h_cov_operator = np.cov(h_cca_training_gaussian.T, rowvar=False)  # same

        # Evaluate the covariance in d (here we assume no data error, so C is identity times a given factor)
        x_dim = np.size(d_pc_training, axis=1)  # Number of PCA components for the curves
        noise = .01
        d_cov_operator = np.eye(x_dim) * noise  # I matrix.
        d_noise_covariance = d_rotations.T @ d_cov_operator @ d_rotations  # same

        # Linear modeling d to h
        # Transpose to get same as Thomas
        g = np.linalg.lstsq(h_cca_training_gaussian.T, d_cca_training.T, rcond=None)[0].T
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

        self.posterior_mean = h_mean_posterior.T[0]
        self.posterior_covariance = h_posterior_covariance

        return h_mean_posterior.T[0], h_posterior_covariance

    def random_sample(self, sample_n, pca_d, pca_h, cca_obj, n_posts=1, add_comp=0):
        # Cut desired number of PC components
        d_pc_training, d_pc_prediction = pca_d.pca_refresh(pca_d.ncomp)
        h_pc_training, h_pc_prediction = pca_h.pca_refresh(pca_h.ncomp)

        d_pc_obs = d_pc_prediction[sample_n]  # data for prediction sample
        h_pc_obs = h_pc_prediction[sample_n]  # target for prediction sample

        d_cca_training, h_cca_training = cca_obj.transform(d_pc_training, h_pc_training)
        d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T

        # Ensure Gaussian distribution in h_cca
        h_cca_training_gaussian = self.ops.gaussian_distribution(h_cca_training)

        # Get the rotation matrices
        d_rotations = cca_obj.x_rotations_

        # Project observed data into canonical space.
        d_cca_prediction, _ = cca_obj.transform(d_pc_obs.reshape(1, -1), h_pc_obs.reshape(1, -1))
        d_cca_prediction = d_cca_prediction.T

        # Estimate the posterior mean and covariance (Tarantola)
        h_mean_posterior, h_posterior_covariance = self.posterior(h_cca_training_gaussian,
                                                                  d_cca_training,
                                                                  d_pc_training,
                                                                  d_rotations,
                                                                  d_cca_prediction)
        shp = pca_h.raw_data.shape  # Original shape
        # n_posts = 500  # Number of estimates sampled from the distribution.
        # Draw n_posts random samples from the multivariate normal distribution :
        h_posts_gaussian = np.random.multivariate_normal(mean=self.posterior_mean,
                                                         cov=self.posterior_covariance,
                                                         size=n_posts).T
        # This h_posts gaussian need to be inverse-transformed to the original distribution.
        # We get the CCA scores.
        h_posts = self.ops.gaussian_inverse(h_posts_gaussian)
        # Calculate the values of hf, i.e. reverse the canonical correlation, it always works if dimf > dimh
        # The value of h_pca_reverse are the score of PCA in the forecast space.
        # To reverse data in the original space, perform the matrix multiplication between the data in the CCA space
        # with the y_loadings matrix. Because CCA scales the input, we must multiply the output by the y_std dev
        # and add the y_mean.
        h_pca_reverse = np.matmul(h_posts.T, cca_obj.y_loadings_.T) * cca_obj.y_std_ + cca_obj.y_mean_

        # Whether to add or not the rest of PC components
        if add_comp:
            rnpc = np.array([pca_h.pc_random(n_posts) for i in range(n_posts)])
            h_pca_reverse = np.array([np.concatenate((h_pca_reverse[i], rnpc[i])) for i in range(n_posts)])
        # Generate forecast in the initial dimension and reshape.
        forecast_ = pca_h.inverse_transform(h_pca_reverse).reshape((n_posts, shp[1], shp[2]))

        return forecast_
