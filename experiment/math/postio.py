#  Copyright (c) 2020. Robin Thibaut, Ghent University

from os.path import join as jp

import joblib
import numpy as np

from experiment.processing.predictions import TargetIO


class PosteriorIO:

    def __init__(self, directory=None):
        self.posterior_mean = None
        self.posterior_covariance = None
        self.seed = None
        self.ops = TargetIO()
        self.directory = directory

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
        if isinstance(h_cca_training_gaussian, (list, tuple, np.ndarray)):
            shctg = np.shape(h_cca_training_gaussian)  # Shape = (n_components_CCA, n_training)
        if isinstance(d_cca_training, (list, tuple, np.ndarray)):
            sdct = np.shape(d_cca_training)  # Shape = (n_components_CCA, n_training)
        if isinstance(d_pc_training, (list, tuple, np.ndarray)):
            sdpt = np.shape(d_pc_training)  # Shape = (n_training, n_components_PCA)
        if isinstance(d_rotations, (list, tuple, np.ndarray)):
            sdr = np.shape(d_rotations)  # Shape = (n_components_PCA, n_components_CCA)
        if isinstance(d_cca_prediction, (list, tuple, np.ndarray)):
            sdcp = np.shape(d_cca_prediction)  # Shape = (n_components_CCA, 1)

        # Size of the set
        n_training = d_cca_training.shape[1]

        # Evaluate the covariance in h (in Canonical space)
        h_cov_operator = np.cov(h_cca_training_gaussian.T, rowvar=False)  # (n_comp_CCA, n_comp_CCA)

        # Evaluate the covariance in d (here we assume no data error, so C is identity times a given factor)
        x_dim = np.size(d_pc_training, axis=1)  # Number of PCA components for the curves
        noise = .01
        d_cov_operator = np.eye(x_dim) * noise  # I matrix. (n_comp_PCA, n_comp_PCA)
        d_noise_covariance = d_rotations.T @ d_cov_operator @ d_rotations  # (n_comp_CCA, n_comp_CCA)

        # Linear modeling d to h (in canonical space) with least-square criterion.
        # Pay attention to the transpose operator.
        g = np.linalg.lstsq(h_cca_training_gaussian.T, d_cca_training.T, rcond=None)[0].T
        # Replace values below threshold by 0.
        g = np.where(np.abs(g) < 1e-12, 0, g)  # (n_comp_CCA, n_comp_CCA)

        # Modeling error due to deviations from theory
        d_ls_predicted = g @ h_cca_training_gaussian  # (n_components_CCA, n_training)
        d_modeling_mean_error = np.mean(d_cca_training - d_ls_predicted, axis=1).reshape(-1, 1)  # (n_comp_CCA, 1)
        d_modeling_error = \
            d_cca_training \
            - d_ls_predicted \
            - np.tile(d_modeling_mean_error, (1, np.size(d_cca_training, axis=1)))
        # (n_comp_CCA, n_training)

        # Information about the covariance of the posterior distribution in Canonical space.
        d_modeling_covariance = (d_modeling_error @ d_modeling_error.T) / n_training  # (n_comp_CCA, n_comp_CCA)

        # Computation of the posterior mean in Canonical space
        h_mean = np.row_stack(np.mean(h_cca_training_gaussian, axis=1))  # (n_comp_CCA, 1)
        h_mean = np.where(np.abs(h_mean) < 1e-12, 0, h_mean)  # My mean is 0, as expected.

        # Equations from Tarantola:
        # h posterior mean (Canonical space)
        h_mean_posterior = \
            h_mean \
            + h_cov_operator @ g.T \
            @ np.linalg.pinv(g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance) \
            @ (d_cca_prediction.reshape(-1, 1) + d_modeling_mean_error - g @ h_mean)  # (n_comp_CCA, 1)

        # h posterior covariance (Canonical space)
        h_posterior_covariance = \
            h_cov_operator \
            - (h_cov_operator @ g.T) \
            @ np.linalg.pinv(g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance) \
            @ g @ h_cov_operator

        h_posterior_covariance = (h_posterior_covariance + h_posterior_covariance.T) / 2  # (n_comp_CCA, n_comp_CCA)

        self.posterior_mean = h_mean_posterior.T[0]  # (n_comp_CCA,)
        self.posterior_covariance = h_posterior_covariance  # (n_comp_CCA, n_comp_CCA)

    def back_transform(self, h_posts_gaussian, cca_obj, pca_h, n_posts, add_comp=False, save_target_pc=False):
        """

        :param h_posts_gaussian:
        :param cca_obj:
        :param pca_h:
        :param n_posts:
        :param add_comp:
        :param save_target_pc:
        :return:
        """
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
        if add_comp:  # TODO: double check
            rnpc = np.array([pca_h.pc_random(n_posts) for _ in range(n_posts)])  # Get the extra components
            h_pca_reverse = np.array([np.concatenate((h_pca_reverse[i], rnpc[i])) for i in range(n_posts)])  # Insert it

        if save_target_pc:
            fname = jp(self.directory, 'target_pc.npy')
            np.save(fname, h_pca_reverse)

        # Generate forecast in the initial dimension and reshape.
        forecast_posterior = \
            pca_h.inverse_transform(h_pca_reverse).reshape((n_posts,
                                                            pca_h.training_shape[1],
                                                            pca_h.training_shape[2]))

        return forecast_posterior

    def random_sample(self, pca_d, pca_h, cca_obj, n_posts, add_comp=False):
        """

        :param pca_d: PCA object for observation
        :param pca_h: PCA object for target
        :param cca_obj: CCA object
        :param n_posts: Number of posteriors to extract
        :param add_comp: Flag to add remaining components
        :return:
        """

        if self.posterior_mean is None and self.posterior_covariance is None:
            # Cut desired number of PC components
            d_pc_training, d_pc_prediction = pca_d.pca_refresh(pca_d.ncomp)
            h_pc_training, _ = pca_h.pca_refresh(pca_h.ncomp)

            d_pc_obs = d_pc_prediction[0]  # observation data for prediction sample

            d_cca_training, h_cca_training = cca_obj.transform(d_pc_training, h_pc_training)
            d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T

            # Ensure Gaussian distribution in h_cca
            h_cca_training_gaussian = self.ops.gaussian_distribution(h_cca_training)

            # Get the rotation matrices
            d_rotations = cca_obj.x_rotations_

            # Project observed data into canonical space.
            d_cca_prediction = cca_obj.transform(d_pc_obs.reshape(1, -1))
            d_cca_prediction = d_cca_prediction.T

            # Estimate the posterior mean and covariance (Tarantola)
            self.posterior(h_cca_training_gaussian,
                           d_cca_training,
                           d_pc_training,
                           d_rotations,
                           d_cca_prediction)

            # Set the seed for later use
            if self.seed is None:
                self.seed = np.random.randint(1e12)
            np.random.seed(self.seed)

            joblib.dump(self, jp(self.directory, 'post.pkl'))

        # Draw n_posts random samples from the multivariate normal distribution :
        # Pay attention to the transpose operator
        h_posts_gaussian = np.random.multivariate_normal(mean=self.posterior_mean,
                                                         cov=self.posterior_covariance,
                                                         size=n_posts).T

        # Back-transform
        forecast_posterior = self.back_transform(h_posts_gaussian=h_posts_gaussian,
                                                 cca_obj=cca_obj,
                                                 pca_h=pca_h,
                                                 n_posts=n_posts,
                                                 add_comp=add_comp)

        return forecast_posterior
