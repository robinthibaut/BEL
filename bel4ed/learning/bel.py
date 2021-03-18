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

from ..config import Setup

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
        self._x_shape, self._y_shape = None, None
        self._x, self._y = None, None
        # Dataset after preprocessing
        self._x_pc, self._y_pc = None, None
        # Dataset after learning
        self._x_c, self._y_c = None, None
        # Dataset after postprocessing
        self._x_f, self._y_f = None, None
        # Observation data
        self._x_obs = None
        self._y_obs = None

    def fit(self, X, Y):
        """
        Fit all pipelines
        :param X:
        :param Y:
        :return:
        """
        check_consistent_length(X, Y)
        # Store original shape
        self._x_shape, self._y_shape = X.shape, Y.shape
        X = self._validate_data(
            X, dtype=np.float64, copy=self.copy, ensure_min_samples=2
        )
        Y = check_array(Y, dtype=np.float64, copy=self.copy, ensure_2d=False)

        self._x, self._y = X, Y

        _xt, _yt = (
            self.X_pre_processing.fit_transform(self._x),
            self.Y_pre_processing.fit_transform(self._y),
        )

        _xt, _yt = (
            _xt[:, : Setup.HyperParameters.n_pc_predictor],
            _yt[:, : Setup.HyperParameters.n_pc_target],
        )

        # Dataset after preprocessing
        self._x_pc, self._y_pc = _xt, _yt

        # Canonical variates
        _xc, _yc = self.cca.fit_transform(X=_xt, y=_yt)

        self._x_c, self._y_c = _xc, _yc

        # CV Normalized
        _xf, _yf = (
            self.X_post_processing.fit_transform(_xc),
            self.Y_post_processing.fit_transform(_yc),
        )

        self._x_f, self._y_f = _xf, _yf

        return self

    def transform(self, X=None, Y=None) -> (np.array, np.array):
        """
        Transform all pipelines
        :param X:
        :param Y:
        :return:
        """

        check_is_fitted(self.cca)
        X = check_array(X, copy=True, dtype=FLOAT_DTYPES)

        # The key here is to cut PC's based on the number defined in configuration file

        if X is not None and Y is None:
            X = check_array(X, copy=True)
            _xt = self.X_pre_processing.transform(X)
            _xt = _xt[:, : Setup.HyperParameters.n_pc_predictor]
            _xc = self.cca.transform(X=_xt)
            _xp = self.X_post_processing.transform(_xc)

            return _xp

        elif Y is not None and X is None:
            Y = check_array(Y, copy=True, ensure_2d=False)
            _xt, _yt = (
                self.X_pre_processing.transform(self._x),
                self.Y_pre_processing.transform(Y),
            )
            _xt, _yt = (
                _xt[:, : Setup.HyperParameters.n_pc_predictor],
                _yt[:, : Setup.HyperParameters.n_pc_target],
            )
            _, _yc = self.cca.transform(X=_xt, Y=_yt)
            _yp = self.Y_post_processing.transform(_yc)

            return _yp

        else:
            _xt, _yt = (
                self.X_pre_processing.transform(self._x),
                self.Y_pre_processing.transform(self._y),
            )
            _xt, _yt = (
                _xt[:, : Setup.HyperParameters.n_pc_predictor],
                _yt[:, : Setup.HyperParameters.n_pc_target],
            )
            _xc, _yc = self.cca.transform(X=_xt, Y=_yt)

            _xp, _yp = (
                self.X_post_processing.transform(_xc),
                self.Y_post_processing.transform(_yc),
            )

            return _xp, _yp

    def random_sample(self, n_posts: int = None) -> np.array:
        """
        :param n_posts:
        :return:
        """
        # Set the seed for later use
        if self.seed is None:
            self.seed = np.random.randint(2 ** 32 - 1, dtype="uint32")

        check_is_fitted(self.cca)
        if n_posts is None:
            self.n_posts = Setup.HyperParameters.n_posts
        # Draw n_posts random samples from the multivariate normal distribution :
        # Pay attention to the transpose operator
        np.random.seed(self.seed)
        Y_samples = np.random.multivariate_normal(
            mean=self.posterior_mean, cov=self.posterior_covariance, size=self.n_posts
        )

        return Y_samples

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit-Transform all pipelines
        :param X:
        :param y:
        :return:
        """

        return self.fit(X, y).transform(X, y)

    def predict(self, X_obs) -> (np.array, np.array):
        """
        Make predictions, in the BEL fashion.
        """
        X_obs = check_array(X_obs)
        self._x_obs = X_obs
        # Project observed data into canonical space.
        X_obs = self.X_pre_processing.transform(X_obs)
        X_obs = X_obs[:, : Setup.HyperParameters.n_pc_predictor]
        X_obs = self.cca.transform(X_obs)
        X_obs = self.X_post_processing.transform(X_obs)

        # Evaluate the covariance in d (here we assume no data error, so C is identity times a given factor)
        # Number of PCA components for the curves
        x_dim = Setup.HyperParameters.n_pc_predictor
        noise = 0.01
        # I matrix. (n_comp_PCA, n_comp_PCA)
        x_cov = np.eye(x_dim) * noise
        # (n_comp_CCA, n_comp_CCA)
        # Get the rotation matrices
        x_rotations = self.cca.x_rotations_
        x_cov = x_rotations.T @ x_cov @ x_rotations
        dict_args = {"x_cov": x_cov}

        X, Y = self._x_f, self._y_f
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
        Y_pred = check_array(Y_pred, dtype=FLOAT_DTYPES)

        y_post = self.Y_post_processing.inverse_transform(Y_pred)
        y_post = (
            np.matmul(y_post, self.cca.y_loadings_.T) * self.cca.y_std_
            + self.cca.y_mean_
        )

        nc = self.Y_pre_processing["pca"].n_components_
        dummy = np.zeros((nc, nc))
        dummy[:, : y_post.shape[1]] = y_post
        y_post = self.Y_pre_processing.inverse_transform(dummy)

        return y_post
