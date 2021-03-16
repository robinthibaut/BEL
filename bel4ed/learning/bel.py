#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
import warnings
from os.path import join as jp
from typing import Type

import joblib
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from loguru import logger

from .. import utils
from ..utils import Root
from ..config import Setup

from ..algorithms import signed_distance
from ..algorithms import mvn_inference
from ..spatial import grid_parameters
from ..processing import PC


def base_pca(
    base: Type[Setup],
    base_dir: str,
    training_roots: Root,
    test_roots: Root,
    h_pca_obj_path: str = None,
):
    """
    Initiate BEL by performing PCA on the training targets or features.
    :param base: class: Base class object
    :param base_dir: str: Base directory path
    :param training_roots: list:
    :param test_roots: list:
    :param h_pca_obj_path:
    :return:
    """

    x_lim, y_lim, grf = base.Focus.x_range, base.Focus.y_range, base.Focus.cell_dim

    if h_pca_obj_path is not None:
        # Loads the results:
        _, pzs_training, r_training_ids = utils.data_loader(
            roots=training_roots, h=True
        )
        _, pzs_test, r_test_ids = utils.data_loader(roots=test_roots, h=True)

        # Load parameters:
        xys, nrow, ncol = grid_parameters(
            x_lim=x_lim, y_lim=y_lim, grf=grf
        )  # Initiate SD instance

        # PCA on signed distance
        # Compute signed distance on pzs.
        # h is the matrix of target feature on which PCA will be performed.
        h_training = np.array(
            [signed_distance(xys, nrow, ncol, grf, pp) for pp in pzs_training]
        )
        h_test = np.array(
            [signed_distance(xys, nrow, ncol, grf, pp) for pp in pzs_test]
        )

        # Convert to dataframes
        training_df_target = utils.i_am_framed(array=h_training, ids=training_roots)
        test_df_target = utils.i_am_framed(array=h_test, ids=test_roots)

        # Initiate h pca object
        h_pco = PC(
            name="h",
            training_df=training_df_target,
            test_df=test_df_target,
            directory=base_dir,
        )
        # Transform
        h_pco.training_fit_transform()
        # Define number of components to keep
        h_pco.n_pc_cut = base.HyperParameters.n_pc_target
        # Transform test arrays
        h_pco.test_transform()

        # Dump
        joblib.dump(h_pco, h_pca_obj_path)

        # Save roots id's in a dat file
        if not os.path.exists(jp(base_dir, "roots.dat")):
            with open(jp(base_dir, "roots.dat"), "w") as f:
                for (
                    r_training_ids
                ) in training_roots:  # Saves roots name until test roots
                    f.write(os.path.basename(r_training_ids) + "\n")

        # Save roots id's in a dat file
        if not os.path.exists(jp(base_dir, "test_roots.dat")):
            with open(jp(base_dir, "test_roots.dat"), "w") as f:
                for r_training_ids in test_roots:  # Saves roots name until test roots
                    f.write(os.path.basename(r_training_ids) + "\n")

        return h_pco

    else:
        logger.error("No base dimensionality reduction could be performed")


class BEL(BaseEstimator):
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
    ):

        self.mode = mode

        self.X_pre_processing = X_pre_processing
        self.Y_pre_processing = Y_pre_processing
        self.X_post_processing = X_post_processing
        self.Y_post_processing = Y_post_processing
        self.cca = cca

        self.posterior_mean = None
        self.posterior_covariance = None

        self.seed = None
        self.n_posts = None

        self._x, self._y = None, None  # Original dataset

    def fit(self, X, Y):
        """
        Fit all pipelines
        :param X:
        :param Y:
        :return:
        """
        X = check_array(X, copy=True, ensure_2d=False)
        Y = check_array(Y, copy=True, ensure_2d=False)

        if self._x is None and self._y is None:
            self._x, self._y = X, Y

        _xt, _yt = (
            self.X_pre_processing.fit_transform(self._x),
            self.Y_pre_processing.fit_transform(self._y),
        )

        _xt, _yt = (
            _xt[:, : Setup.HyperParameters.n_pc_predictor],
            _yt[:, : Setup.HyperParameters.n_pc_target],
        )

        _xc, _yc = self.cca.fit_transform(X=_xt, y=_yt)

        self.X_post_processing.fit(_xc), self.Y_post_processing.fit(_yc)

    def transform(self, X=None, Y=None):
        """
        Transform all pipelines
        :param X:
        :param Y:
        :return:
        """

        check_is_fitted(self.cca)

        if X is not None and Y is None:
            X = check_array(X, copy=True, ensure_2d=False)
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

    def _random_sample(self, n_posts: int = None) -> np.array:
        """
        :param n_posts:
        :return:
        """
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

    def fit_transform(self, X, Y) -> (np.array, np.array):
        """
        Fit-Transform all pipelines
        :param X:
        :param Y:
        :return:
        """

        X = check_array(X, copy=True, ensure_2d=False)
        Y = check_array(Y, copy=True, ensure_2d=False)

        _xt, _yt = (
            self.X_pre_processing.fit_transform(X),
            self.Y_pre_processing.fit_transform(Y),
        )

        _xt, _yt = (
            _xt[:, : Setup.HyperParameters.n_pc_predictor],
            _yt[:, : Setup.HyperParameters.n_pc_target],
        )

        _xc, _yc = self.cca.fit_transform(X=_xt, y=_yt)

        _xp, _yp = self.X_post_processing.fit(_xc), self.Y_post_processing.fit(_yc)

        return _xp, _yp

    def predict(self, X_obs) -> np.array:
        """
        Make predictions, in the BEL fashion.
        """
        X_obs = check_array(X_obs)
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

        X, Y = self.transform()
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

        # Set the seed for later use
        if self.seed is None:
            self.seed = np.random.randint(2 ** 32 - 1, dtype="uint32")

        # Sample the inferred multivariate gaussian distribution
        random_samples = self._random_sample(self.n_posts)

        return random_samples

    def inverse_transform(
        self,
        Y_pred,
    ) -> np.array:
        """
        Back-transforms the sampled gaussian distributed posterior Y to their physical space.
        :param Y_pred:
        :return: forecast_posterior
        """
        y_post = self.Y_post_processing.inverse_transform(Y_pred)
        y_post = (
            np.matmul(y_post, self.cca.y_loadings_.T) * self.cca.y_std_
            + self.cca.y_mean_
        )

        nc = self.Y_pre_processing["pca"].n_components_
        dummy = np.zeros(nc)
        dummy[: y_post.shape[1]] = y_post
        y_post = self.Y_pre_processing.inverse_transform(dummy)

        return y_post

