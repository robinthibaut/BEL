#  Copyright (c) 2021. Robin Thibaut, Ghent University

import warnings

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import (
    check_is_fitted,
    check_consistent_length,
    FLOAT_DTYPES,
)
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    MultiOutputMixin,
)


from ..algorithms import mvn_inference


class BEL(TransformerMixin, MultiOutputMixin, BaseEstimator):
    """
    Heart of the framework.
    """

    def __init__(
        self,
        mode: str = "mvn",
        X_pre_processing=None,
        Y_pre_processing=None,
        X_post_processing=None,
        Y_post_processing=None,
        cca=None,
        copy=True,
    ):
        self.copy = copy
        # How to infer the posterior parameters
        self.mode = mode

        # Processing pipelines
        self.X_pre_processing = X_pre_processing
        self.Y_pre_processing = Y_pre_processing
        self.X_post_processing = X_post_processing
        self.Y_post_processing = Y_post_processing
        self.cca = cca

        # Posterior parameters
        self.posterior_mean = None
        self.posterior_covariance = None

        # Parameters for sampling
        self.seed = None
        self.n_posts = None

        # Original dataset
        self.X_shape, self.Y_shape = None, None
        self.X, self.Y = None, None
        self.X_obs, self.Y_obs = None, None  # Observation data

        # Dataset after preprocessing (dimension-reduced as indicated in config)
        self. X_n_pc, self.Y_n_pc = None, None
        self.X_pc, self.Y_pc = None, None
        self.X_obs_pc, self.Y_obs_pc = None, None
        # Dataset after learning
        self.X_c, self.Y_c = None, None
        self.X_obs_c, self.Y_obs_c = None, None
        # Dataset after postprocessing
        self.X_f, self.Y_f = None, None
        self.X_obs_f, self.Y_obs_f = None, None

    def fit(self, X, Y):
        """
        Fit all pipelines
        :param X:
        :param Y:
        :return:
        """
        check_consistent_length(X, Y)
        self.X, self.Y = X, Y  # Save dataframe with names
        # Store original shape
        self.X_shape, self.Y_shape = (
            X.attrs["physical_shape"],
            Y.attrs["physical_shape"],
        )
        _X = self._validate_data(
            X, dtype=np.float64, copy=self.copy, ensure_min_samples=2
        )
        _Y = check_array(Y, dtype=np.float64, copy=self.copy, ensure_2d=False)

        _xt, _yt = (
            self.X_pre_processing.fit_transform(_X),
            self.Y_pre_processing.fit_transform(_Y),
        )

        _xt, _yt = (
            _xt[:, : self.X_n_pc],
            _yt[:, : self.Y_n_pc],
        )  # Cut PC

        # Dataset after preprocessing
        self.X_pc, self.Y_pc = _xt, _yt

        # Canonical variates
        _xc, _yc = self.cca.fit_transform(X=_xt, y=_yt)

        self.X_c, self.Y_c = _xc, _yc

        # CV Normalized
        _xf, _yf = (
            self.X_post_processing.fit_transform(_xc),
            self.Y_post_processing.fit_transform(_yc),
        )

        self.X_f, self.Y_f = _xf, _yf

        return self

    def transform(self, X=None, Y=None) -> (np.array, np.array):
        """
        Transform data across all pipelines
        :param X:
        :param Y:
        :return:
        """

        check_is_fitted(self.cca)
        # The key here is to cut PC's based on the number defined in configuration file

        if X is not None and Y is None:
            X = check_array(X, copy=True)
            _xt = self.X_pre_processing.transform(X)
            _xt = _xt[:, : self.X_n_pc]
            _xc = self.cca.transform(X=_xt)
            _xp = self.X_post_processing.transform(_xc)

            return _xp

        elif Y is not None and X is None:
            Y = check_array(Y, copy=True, ensure_2d=False)
            _xt, _yt = (
                self.X_pre_processing.transform(self.X),
                self.Y_pre_processing.transform(Y),
            )
            _xt, _yt = (
                _xt[:, : self.X_n_pc],
                _yt[:, : self.Y_n_pc],
            )
            _, _yc = self.cca.transform(X=_xt, Y=_yt)
            _yp = self.Y_post_processing.transform(_yc)

            return _yp

        else:
            _xt, _yt = (
                self.X_pre_processing.transform(self.X),
                self.Y_pre_processing.transform(self.Y),
            )
            _xt, _yt = (
                _xt[:, : self.X_n_pc],
                _yt[:, : self.Y_n_pc],
            )
            _xc, _yc = self.cca.transform(X=_xt, Y=_yt)

            _xp, _yp = (
                self.X_post_processing.transform(_xc),
                self.Y_post_processing.transform(_yc),
            )

            return _xp, _yp

    def random_sample(self, n_posts: int = None) -> np.array:
        """
        Random sample the inferred posterior Gaussian distribution
        :param n_posts:
        :return:
        """
        # Set the seed for later use
        if self.seed is None:
            self.seed = np.random.randint(2 ** 32 - 1, dtype="uint32")

        check_is_fitted(self.cca)
        if n_posts is None:
            n_posts = self.n_posts
        # Draw n_posts random samples from the multivariate normal distribution :
        # Pay attention to the transpose operator
        np.random.seed(self.seed)
        Y_samples = np.random.multivariate_normal(
            mean=self.posterior_mean, cov=self.posterior_covariance, size=n_posts
        )

        return Y_samples

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit-Transform accross all pipelines
        :param X:
        :param y:
        :return:
        """

        return self.fit(X, y).transform(X, y)

    def predict(self, X_obs) -> (np.array, np.array):
        """
        Make predictions, in the BEL fashion.
        Maybe rename to "fit" since the "predictions" come with "random_sample"
        """
        self.X_obs = X_obs  # Save dataframe with name
        X_obs = check_array(self.X_obs)
        # Project observed data into canonical space.
        X_obs = self.X_pre_processing.transform(X_obs)
        X_obs = X_obs[:, : self.X_n_pc]
        self.X_obs_pc = X_obs
        X_obs = self.cca.transform(X_obs)
        self.X_obs_c = X_obs
        X_obs = self.X_post_processing.transform(X_obs)
        self.X_obs_f = X_obs

        # Evaluate the covariance in d (here we assume no data error, so C is identity times a given factor)
        # Number of PCA components for the curves
        x_dim = self.X_n_pc
        noise = 0.01
        # I matrix. (n_comp_PCA, n_comp_PCA)
        x_cov = np.eye(x_dim) * noise
        # (n_comp_CCA, n_comp_CCA)
        # Get the rotation matrices
        x_rotations = self.cca.x_rotations_
        x_cov = x_rotations.T @ x_cov @ x_rotations
        dict_args = {"x_cov": x_cov}

        X, Y = self.X_f, self.Y_f
        # Estimate the posterior mean and covariance
        if self.mode == "mvn":
            self.posterior_mean, self.posterior_covariance = mvn_inference(
                X=X,
                Y=Y,
                X_obs=X_obs,
                **dict_args,
            )
        else:
            warnings.warn("KDE not implemented yet")

        return self.posterior_mean, self.posterior_covariance

    def inverse_transform(
        self,
        Y_pred,
    ) -> np.array:
        """
        Back-transforms the sampled gaussian distributed posterior Y to their physical space.
        :param Y_pred:
        :return: forecast_posterior
        """
        check_is_fitted(self.cca)
        Y_pred = check_array(Y_pred)

        y_post = self.Y_post_processing.inverse_transform(
            Y_pred
        )  # Posterior CCA scores
        y_post = (
            np.matmul(y_post, self.cca.y_loadings_.T) * self.cca.y_std_
            + self.cca.y_mean_
        )  # Posterior PC scores

        # Back transform PC scores
        nc = self.Y_pre_processing["pca"].n_components_  # Number of components
        dummy = np.zeros((self.n_posts, nc))  # Create a dummy matrix filled with zeros
        dummy[
            :, : y_post.shape[1]
        ] = y_post  # Fill the dummy matrix with the posterior PC
        y_post = self.Y_pre_processing.inverse_transform(dummy)  # Inverse transform

        return y_post
