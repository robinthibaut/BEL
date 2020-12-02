#  Copyright (c) 2020. Robin Thibaut, Ghent University

from os.path import join as jp

import joblib
import numpy as np

from sklearn.preprocessing import PowerTransformer

# Try Gaussian Process from sklearn (however does not support multiple output)
# import sklearn.gaussian_process as gp


# import gpflow as gpf
# import tensorflow as tf
# from gpflow.utilities import print_summary
# from gpflow.ci_utils import ci_niter

from experiment.base.inventory import MySetup
from experiment.processing.target import TargetIO


class PosteriorIO:

    def __init__(self, directory: str = None):
        self.posterior_mean = None
        self.posterior_covariance = None
        self.seed = None
        self.n_posts = None
        self.processing = TargetIO()
        self.test_h = PowerTransformer(method='yeo-johnson', standardize=True)
        self.test_d = PowerTransformer(method='yeo-johnson', standardize=True)
        self.directory = directory

    def linear_gaussian_regression(self,
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
        :return: h_mean_posterior, h_posterior_covariance
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

        # Evaluate the covariance in h (in Canonical space)
        h_cov_operator = np.cov(h_cca_training_gaussian, rowvar=False)  # (n_comp_CCA, n_comp_CCA)

        # Evaluate the covariance in d (here we assume no data error, so C is identity times a given factor)
        x_dim = np.size(d_pc_training, axis=1)  # Number of PCA components for the curves
        noise = .01
        d_cov_operator = np.eye(x_dim) * noise  # I matrix. (n_comp_PCA, n_comp_PCA)
        d_noise_covariance = d_rotations.T @ d_cov_operator @ d_rotations  # (n_comp_CCA, n_comp_CCA)

        # Linear modeling h to d (in canonical space) with least-square criterion.
        # Pay attention to the transpose operator.
        # Computes the vector g that approximatively solves the equation h @ g = d.
        g = np.linalg.lstsq(h_cca_training_gaussian, d_cca_training, rcond=None)[0]
        # Replace values below threshold by 0.
        g = np.where(np.abs(g) < 1e-12, 0, g)  # (n_comp_CCA, n_comp_CCA)

        # Modeling error due to deviations from theory
        d_ls_predicted = h_cca_training_gaussian @ g  # (n_components_CCA, n_training)
        d_modeling_mean_error = np.mean(d_cca_training - d_ls_predicted, axis=0).reshape(1, -1)  # (n_comp_CCA, 1)
        d_modeling_error = \
            d_cca_training \
            - d_ls_predicted \
            - np.tile(d_modeling_mean_error, (n_training, 1))
        # (n_comp_CCA, n_training)

        # Information about the covariance of the posterior distribution in Canonical space.
        # d_modeling_covariance = (d_modeling_error.T @ d_modeling_error) / (n_training-1)  # (n_comp_CCA, n_comp_CCA)
        d_modeling_covariance = np.cov(d_modeling_error.T)  # (n_comp_CCA, n_comp_CCA)

        # Computation of the posterior mean in Canonical space
        h_mean = np.column_stack(np.mean(h_cca_training_gaussian, axis=0))  # (n_comp_CCA, 1)
        h_mean = np.where(np.abs(h_mean) < 1e-12, 0, h_mean)  # My mean is 0, as expected.

        # Equations from Tarantola:
        # h posterior mean (Canonical space)
        h_mean_posterior = \
            h_mean.T \
            + h_cov_operator @ g \
            @ np.linalg.pinv(g.T @ h_cov_operator @ g + d_noise_covariance + d_modeling_covariance) \
            @ (d_cca_prediction + d_modeling_mean_error - h_mean @ g).T  # (n_comp_CCA, 1)

        # h posterior covariance (Canonical space)
        h_posterior_covariance = \
            h_cov_operator \
            - (h_cov_operator @ g) \
            @ np.linalg.pinv(g.T @ h_cov_operator @ g + d_noise_covariance + d_modeling_covariance) \
            @ g.T @ h_cov_operator

        h_posterior_covariance = (h_posterior_covariance + h_posterior_covariance.T) / 2  # (n_comp_CCA, n_comp_CCA)

        self.posterior_mean = h_mean_posterior.T[0]  # (n_comp_CCA,)
        self.posterior_covariance = h_posterior_covariance  # (n_comp_CCA, n_comp_CCA)

    # def gaussian_process_regression(self, X_tr, y_tr, X_te):
    #     # GPR does not work with multiple dimensions output !
    #     kernel = gp.kernels.RBF()
    #     model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.01, normalize_y=True)
    #     model.fit(X_tr, y_tr)
    #     params = model.kernel_.get_params()
    #     y_pred, cov = model.predict(X_te, return_cov=True)
    #     print(cov.shape)
    #     self.posterior_mean = y_pred[0]
    #     self.posterior_covariance = cov[0]

    # def MOGPR(self, X, Y, test):
    #     data = X, Y
    #     P = Y.shape[1]
    #     M = 20  # Number of inducing points
    #     idr = np.random.choice(np.arange(X.shape[1]), M)
    #     Zinit = X[idr]
    #
    #     # create multi-output kernel
    #     kernel = gpf.kernels.SharedIndependent(
    #         gpf.kernels.SquaredExponential() + gpf.kernels.Linear(), output_dim=P
    #     )
    #     # initialization of inducing input locations (M random points from the training inputs)
    #     Z = Zinit.copy()
    #     # create multi-output inducing variables from Z
    #     iv = gpf.inducing_variables.SharedIndependentInducingVariables(
    #         gpf.inducing_variables.InducingPoints(Z)
    #     )
    #
    #     m = gpf.models.SVGP(kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=P)
    #
    #     def optimize_model_with_scipy(model):
    #         optimizer = gpf.optimizers.Scipy()
    #         optimizer.minimize(
    #             model.training_loss_closure(data),
    #             variables=model.trainable_variables,
    #             method="l-bfgs-b",
    #             options={"disp": True, "maxiter": 10000},
    #         )
    #
    #     optimize_model_with_scipy(m)
    #     print_summary(m)
    #
    #     op1, op2 = m.predict_f(test, full_output_cov=True)
    #
    #     self.posterior_mean = op1[0]
    #     self.posterior_covariance = op2[0]

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
        h_posts = self.test_h.inverse_transform(h_posts_gaussian.T)  # (n_components, n_samples)

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
                                                         size=n_posts).T
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
            d_cca_training = self.test_d.fit_transform(d_cca_training)

            # Ensure Gaussian distribution in h_cca_training
            # h_cca_training_gaussian = self.processing.gaussian_distribution(h_cca_training)
            h_cca_training = self.test_h.fit_transform(h_cca_training)

            # Get the rotation matrices
            d_rotations = cca_obj.x_rotations_

            # Project observed data into canonical space.
            d_cca_prediction = cca_obj.transform(d_pc_obs.reshape(1, -1))
            # d_cca_prediction = d_cca_prediction.T
            d_cca_prediction = self.test_d.transform(d_cca_prediction)

            # Estimate the posterior mean and covariance (Tarantola)

            self.linear_gaussian_regression(h_cca_training,
                                            d_cca_training,
                                            d_pc_training,
                                            d_rotations,
                                            d_cca_prediction)

            # Test using sklearn built-in GPR method
            # self.gaussian_process_regression(d_cca_training.T, h_cca_training_gaussian.T, d_cca_prediction.T)
            
            # self.MOGPR(d_cca_training.T, h_cca_training_gaussian.T, d_cca_prediction.T)
            
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
