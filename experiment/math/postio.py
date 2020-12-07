#  Copyright (c) 2020. Robin Thibaut, Ghent University

from os.path import join as jp

import joblib
import numpy as np

from sklearn.preprocessing import PowerTransformer

from experiment.base.inventory import MySetup


class PosteriorIO:

    def __init__(self, directory: str = None):
        self.posterior_mean = None
        self.posterior_covariance = None
        self.seed = None
        self.n_posts = None
        self.normalize_h = PowerTransformer(method='yeo-johnson', standardize=True)
        self.normalize_d = PowerTransformer(method='yeo-johnson', standardize=True)
        self.directory = directory

    def mvn_inference(self,
                      h_cca_training_gaussian,
                      d_cca_training,
                      d_pc_training,
                      d_rotations,
                      d_cca_prediction):
        """
        Estimating posterior mean and covariance of the target.
        .. [1] A. Tarantola. Inverse Problem Theory and Methods for Model Parameter Estimation.
               SIAM, 2005. Pages: 70-71
        :param h_cca_training_gaussian: Canonical Variate of the training target, gaussian-distributed
        :param d_cca_training: Canonical Variate of the training data
        :param d_pc_training: Principal Components of the training data
        :param d_rotations: CCA rotations of the training data (project original data to canonical space)
        :param d_cca_prediction: Canonical Variate of the observation
        :return: h_posterior_mean, h_posterior_covariance
        :raise ValueError: An exception is thrown if the shape of input arrays are not consistent.
        """

        # TODO: add dimension check
        if isinstance(h_cca_training_gaussian, (list, tuple, np.ndarray)):
            shctg = np.shape(h_cca_training_gaussian)  # Shape = (n_components_CCA, n_training)
        if isinstance(d_cca_training, (list, tuple, np.ndarray)):
            sdct = np.shape(d_cca_training)  # Shape = (n_components_CCA, n_training)
        if isinstance(d_pc_training, (list, tuple, np.ndarray)):
            sdpt = np.shape(d_pc_training)  # Shape = (n_training, n_components_PCA)
        if isinstance(d_rotations, (list, tuple, np.ndarray)):
            sdr = np.shape(d_rotations)  # Shape = (n_components_PCA_d, n_components_CCA_h)
        if isinstance(d_cca_prediction, (list, tuple, np.ndarray)):
            sdcp = np.shape(d_cca_prediction)  # Shape = (n_components_CCA, 1)

        # Size of the set
        n_training = d_cca_training.shape[0]

        # Computation of the posterior mean in Canonical space
        h_mean = np.mean(h_cca_training_gaussian, axis=0)  # (n_comp_CCA, 1)
        h_mean = np.where(np.abs(h_mean) < 1e-12, 0, h_mean)  # Mean is 0, as expected.

        # Evaluate the covariance in h (in Canonical space)
        # Very close to the Identity matrix
        h_cov_operator = np.cov(h_cca_training_gaussian.T)  # (n_comp_CCA, n_comp_CCA)

        # Evaluate the covariance in d (here we assume no data error, so C is identity times a given factor)
        x_dim = np.size(d_pc_training, axis=1)  # Number of PCA components for the curves
        noise = .01
        d_cov_operator = np.eye(x_dim) * noise  # I matrix. (n_comp_PCA, n_comp_PCA)
        d_noise_covariance = d_rotations.T @ d_cov_operator @ d_rotations  # (n_comp_CCA, n_comp_CCA)

        # Linear modeling h to d (in canonical space) with least-square criterion.
        # Pay attention to the transpose operator.
        # Computes the vector g that approximatively solves the equation h @ g = d.
        g = np.linalg.lstsq(h_cca_training_gaussian, d_cca_training, rcond=None)[0].T
        # Replace values below threshold by 0.
        g = np.where(np.abs(g) < 1e-12, 0, g)  # (n_comp_CCA, n_comp_CCA)

        # Modeling error due to deviations from theory
        d_ls_predicted = h_cca_training_gaussian @ g.T  # (n_components_CCA, n_training)
        d_modeling_mean_error = np.mean(d_cca_training - d_ls_predicted, axis=0)  # (n_comp_CCA, 1)
        d_modeling_error = \
            d_cca_training \
            - d_ls_predicted \
            - np.tile(d_modeling_mean_error, (n_training, 1))
        # (n_comp_CCA, n_training)

        # Information about the covariance of the posterior distribution in Canonical space.
        d_modeling_covariance = np.cov(d_modeling_error.T)  # (n_comp_CCA, n_comp_CCA)

        # Build block matrix
        s11 = h_cov_operator
        s12 = h_cov_operator @ g.T
        s21 = g @ h_cov_operator
        s22 = g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance
        block = np.block([[s11, s12], [s21, s22]])

        def get_block(pm, i: int):
            """
            Extracts block from a 2x2 partitioned matrix.
            :param pm: Partitioned matrix
            :param i: Block index
            1 2
            3 4
            :return: Bock #b
            """

            b = pm.shape[0]//2

            if i == 1:
                return pm[:b, :b]
            if i == 2:
                return pm[:b, b:]
            if i == 3:
                return pm[b:, :b]
            if i == 4:
                return pm[b:, b:]
            else:
                return 0

        # Inverse
        delta = np.linalg.pinv(block)
        # Partition block
        d11 = get_block(delta, 1)
        d12 = get_block(delta, 2)

        # Observe that posterior covariance does not depend on observed d.
        h_posterior_covariance = np.linalg.pinv(d11)
        # Computing the posterior mean is simply a linear operation, given precomputed posterior covariance.
        h_posterior_mean = h_posterior_covariance @ \
                           (d11 @ h_mean - d12 @ (d_cca_prediction[0] - d_modeling_mean_error - h_mean @ g.T))

        # test = np.block([[d11, d12], [d21, d22]])
        # plt.matshow(test, cmap='coolwarm')
        # plt.colorbar()
        # plt.show()

        # Also works:
        # Inverse of the sample covariance matrix of d ( Sig dd )
        # ddd_inv = np.linalg.pinv(g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance)
        # h_posterior_covariance = h_cov_operator - \
        #     h_cov_operator @ g.T @ ddd_inv @ g @ h_cov_operator
        #
        # h_posterior_mean = \
        #     h_mean + h_cov_operator @ g.T @ ddd_inv @ (d_cca_prediction[0] - d_modeling_mean_error - h_mean @ g.T)

        # h_posterior_covariance = (h_posterior_covariance + h_posterior_covariance.T) / 2  # (n_comp_CCA, n_comp_CCA)

        self.posterior_mean = h_posterior_mean  # (n_comp_CCA,)
        self.posterior_covariance = h_posterior_covariance  # (n_comp_CCA, n_comp_CCA)

    def back_transform(self,
                       h_posts_gaussian,
                       cca_obj,
                       pca_h,
                       n_posts: int,
                       add_comp: bool = False,
                       save_target_pc: bool = False):
        """
        Back-transforms the sampled gaussian distributed posterior h to their physical space.
        :param h_posts_gaussian:
        :param cca_obj:
        :param pca_h:
        :param n_posts:
        :param add_comp:
        :param save_target_pc:
        :return: forecast_posterior
        """
        # This h_posts gaussian need to be inverse-transformed to the original distribution.
        # We get the CCA scores.

        # h_posts = self.processing.gaussian_inverse(h_posts_gaussian)  # (n_components, n_samples)
        h_posts = self.normalize_h.inverse_transform(h_posts_gaussian)  # (n_components, n_samples)

        # Calculate the values of hf, i.e. reverse the canonical correlation, it always works if dimf > dimh
        # The value of h_pca_reverse are the score of PCA in the forecast space.
        # To reverse data in the original space, perform the matrix multiplication between the data in the CCA space
        # with the y_loadings matrix. Because CCA scales the input, we must multiply the output by the y_std dev
        # and add the y_mean.
        h_pca_reverse = np.matmul(h_posts, cca_obj.y_loadings_.T) * cca_obj.y_std_ + cca_obj.y_mean_

        # Whether to add or not the rest of PC components
        if add_comp:  # TODO: double check
            rnpc = np.array([pca_h.pc_random(n_posts) for _ in range(n_posts)])  # Get the extra components
            h_pca_reverse = np.array([np.concatenate((h_pca_reverse[i], rnpc[i])) for i in range(n_posts)])  # Insert it

        if save_target_pc:
            fname = jp(self.directory, 'target_pc.npy')
            np.save(fname, h_pca_reverse)

        # Generate forecast in the initial dimension and reshape.
        forecast_posterior = \
            pca_h.custom_inverse_transform(h_pca_reverse).reshape((n_posts,
                                                                   pca_h.training_shape[1],
                                                                   pca_h.training_shape[2]))

        return forecast_posterior

    def random_sample(self, n_posts: int = None):
        """

        :param n_posts:
        :return:
        """
        if n_posts is None:
            n_posts = self.n_posts
        # Draw n_posts random samples from the multivariate normal distribution :
        # Pay attention to the transpose operator
        np.random.seed(self.seed)
        h_posts_gaussian = np.random.multivariate_normal(mean=self.posterior_mean,
                                                         cov=self.posterior_covariance,
                                                         size=n_posts)
        return h_posts_gaussian

    def bel_predict(self,
                    pca_d,
                    pca_h,
                    cca_obj,
                    n_posts: int,
                    add_comp: bool = False):
        """
        Make predictions, in the BEL fashion.
        :param pca_d: PCA object for observations.
        :param pca_h: PCA object for targets.
        :param cca_obj: CCA object.
        :param n_posts: Number of posteriors to extract.
        :param add_comp: Flag to add remaining components.
        :return: forecast_posterior
        """

        if self.posterior_mean is None and self.posterior_covariance is None:
            # Cut desired number of PC components
            d_pc_training, d_pc_prediction = pca_d.pca_refresh(pca_d.n_pc_cut)
            h_pc_training, _ = pca_h.pca_refresh(pca_h.n_pc_cut)

            d_pc_obs = d_pc_prediction[0]  # observation data for prediction sample

            # Transform to canonical space
            d_cca_training, h_cca_training = cca_obj.transform(d_pc_training, h_pc_training)
            # d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T

            # Ensure Gaussian distribution in d_cca_training
            d_cca_training = self.normalize_d.fit_transform(d_cca_training)

            # Ensure Gaussian distribution in h_cca_training
            # h_cca_training_gaussian = self.processing.gaussian_distribution(h_cca_training)
            h_cca_training = self.normalize_h.fit_transform(h_cca_training)

            # Get the rotation matrices
            d_rotations = cca_obj.x_rotations_

            # Project observed data into canonical space.
            d_cca_prediction = cca_obj.transform(d_pc_obs.reshape(1, -1))

            d_cca_prediction = self.normalize_d.transform(d_cca_prediction)

            # Estimate the posterior mean and covariance (Tarantola)

            self.mvn_inference(h_cca_training,
                               d_cca_training,
                               d_pc_training,
                               d_rotations,
                               d_cca_prediction)

            # Set the seed for later use
            if self.seed is None:
                self.seed = np.random.randint(2 ** 32 - 1, dtype='uint32')

            if n_posts is None:
                self.n_posts = MySetup.Forecast.n_posts
            else:
                self.n_posts = n_posts

            # Saves this postio object to avoid saving large amounts of 'forecast_posterior'
            # This allows to reload this object later on and resample using the same seed.
            joblib.dump(self, jp(self.directory, 'post.pkl'))

        # Sample the inferred multivariate gaussian distribution
        h_posts_gaussian = self.random_sample(self.n_posts)

        # Back-transform h posterior to the physical space
        forecast_posterior = self.back_transform(h_posts_gaussian=h_posts_gaussian,
                                                 cca_obj=cca_obj,
                                                 pca_h=pca_h,
                                                 n_posts=self.n_posts,
                                                 add_comp=add_comp)

        return forecast_posterior
