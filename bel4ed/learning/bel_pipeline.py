#  Copyright (c) 2021. Robin Thibaut, Ghent University
"""
This script pre-processes the data.

- It subdivides the breakthrough curves into an arbitrary number of steps, as the mt3dms results
do not necessarily share the same time steps - d

- It computes the signed distance field for each particles endpoints file - h
It then perform PCA keeping all components on both d and h.

- Finally, CCA is performed after selecting an appropriate number of PC to keep.

It saves 2 pca objects (d, h) and 1 cca object, according to the project ecosystem.
"""

import os
import warnings
from os.path import join as jp
from typing import Type

import joblib
import numpy as np
from sklearn.utils import check_array
from sklearn.preprocessing import PowerTransformer
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import CCA  # As of version 0.24.1, CCA doesn't work
from loguru import logger

from .. import utils
from ..utils import Root
from ..config import Setup

from ..algorithms import signed_distance  # , CCA
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

    def __init__(self, directory: str = None, mode: str = "mvn", pipeline=None):

        self.directory = directory
        self.mode = mode
        self.pipeline = pipeline
        self.posterior_mean = None
        self.posterior_covariance = None
        self.seed = None
        self.n_posts = None
        self.X_normalizer = PowerTransformer(method="yeo-johnson", standardize=True)
        self.Y_normalizer = PowerTransformer(method="yeo-johnson", standardize=True)

    def fit(self, X, Y):
        """
        """
        X = check_array(X, copy=True, ensure_2d=False)
        Y = check_array(Y, copy=True, ensure_2d=False)

        self.pipeline.fit(X, Y)  # Fit

    def random_sample(self, n_posts: int = None) -> np.array:
        """
        :param n_posts:
        :return:
        """
        if n_posts is None:
            n_posts = self.n_posts
        # Draw n_posts random samples from the multivariate normal distribution :
        # Pay attention to the transpose operator
        np.random.seed(self.seed)
        Y_samples = np.random.multivariate_normal(
            mean=self.posterior_mean, cov=self.posterior_covariance, size=n_posts
        )
        return Y_samples

    def fit_transform(self, X, Y):

        x_scores, y_scores = self.pipeline.fit_transform(X, Y)
        return x_scores, y_scores

    def predict(self, pca_d: PC, pca_h: PC, n_posts: int) -> np.array:
        """
        Make predictions, in the BEL fashion.
        :param pca_d: PCA object for observations.
        :param pca_h: PCA object for targets.
        :param n_posts: Number of posteriors to extract.
        :return: forecast_posterior in original space
        """

        if self.posterior_mean is None and self.posterior_covariance is None:
            # Cut desired number of PC components
            d_pc_training, d_pc_prediction = pca_d.comp_refresh(pca_d.n_pc_cut)
            h_pc_training, _ = pca_h.comp_refresh(pca_h.n_pc_cut)

            # observation data for prediction sample
            d_pc_obs = d_pc_prediction[0]

            # Transform to canonical space
            d_cca_training, h_cca_training = self.fit_transform(
                d_pc_training, h_pc_training
            )

            # Ensure Gaussian distribution in d_cca_training
            d_cca_training = self.Y_normalizer.fit_transform(d_cca_training)

            # Ensure Gaussian distribution in h_cca_training
            h_cca_training = self.X_normalizer.fit_transform(h_cca_training)

            # Project observed data into canonical space.
            d_cca_prediction = self.pipeline.transform(d_pc_obs.reshape(1, -1))
            d_cca_prediction = self.Y_normalizer.transform(d_cca_prediction)

            # Evaluate the covariance in d (here we assume no data error, so C is identity times a given factor)
            # Number of PCA components for the curves
            x_dim = np.size(d_pc_training, axis=1)
            noise = 0.01
            # I matrix. (n_comp_PCA, n_comp_PCA)
            x_cov = np.eye(x_dim) * noise
            # (n_comp_CCA, n_comp_CCA)
            # Get the rotation matrices
            d_rotations = self.pipeline['cca'].x_rotations_
            x_cov = d_rotations.T @ x_cov @ d_rotations
            dict_args = {"x_cov": x_cov}

            # Estimate the posterior mean and covariance
            if self.mode == "mvn":
                self.posterior_mean, self.posterior_covariance = mvn_inference(
                    X=d_cca_training,
                    Y=h_cca_training,
                    X_obs=d_cca_prediction,
                    **dict_args,
                )
            else:
                warnings.warn("KDE not implemented yet")

            # Set the seed for later use
            if self.seed is None:
                self.seed = np.random.randint(2 ** 32 - 1, dtype="uint32")

            if n_posts is None:
                self.n_posts = Setup.HyperParameters.n_posts
            else:
                self.n_posts = n_posts

            # Saves this BEL object to avoid saving large amounts of 'forecast_posterior'
            # This allows to reload this object later on and resample using the same seed.
            post_location = jp(self.directory, "post.pkl")
            logger.info(f"Saved posterior object to {post_location}")
            joblib.dump(self, post_location)

        # Sample the inferred multivariate gaussian distribution
        random_samples = self.random_sample(self.n_posts)

        return random_samples

    def inverse_transform(
            self,
            h_posts_gaussian: np.array,
            cca_obj: CCA,
            pca_h: PC,
            add_comp: bool = False,
            save_target_pc: bool = False,
    ) -> np.array:
        """
        Back-transforms the sampled gaussian distributed posterior h to their physical space.
        :param h_posts_gaussian:
        :param cca_obj:
        :param pca_h:
        :param add_comp:
        :param save_target_pc:
        :return: forecast_posterior
        """
        # This h_posts gaussian need to be inverse-transformed to the original distribution.
        # We get the CCA scores.

        # h_posts = self.processing.gaussian_inverse(h_posts_gaussian)  # (n_components, n_samples)
        h_posts = self.X_normalizer.inverse_transform(
            h_posts_gaussian
        )  # (n_components, n_samples)

        # Calculate the values of hf, i.e. reverse the canonical correlation, it always works if dimf > dimh
        # The value of h_pca_reverse are the score of PCA in the forecast space.
        # To reverse data in the original space, perform the matrix multiplication between the data in the CCA space
        # with the y_loadings matrix. Because CCA scales the input, we must multiply the output by the y_std dev
        # and add the y_mean.
        cca_obj = self.pipeline['cca']
        h_pca_reverse = (
                np.matmul(h_posts, cca_obj.y_loadings_.T) * cca_obj.y_std_ + cca_obj.y_mean_
        )

        # Whether to add or not the rest of PC components
        if add_comp:  # TODO: double check
            rnpc = np.array(
                [pca_h.random_pc(self.n_posts) for _ in range(self.n_posts)]
            )  # Get the extra components
            h_pca_reverse = np.array(
                [
                    np.concatenate((h_pca_reverse[i], rnpc[i]))
                    for i in range(self.n_posts)
                ]
            )  # Insert it

        if save_target_pc:
            fname = jp(self.directory, "target_pc.npy")
            np.save(fname, h_pca_reverse)

        osx, osy = (
            pca_h.training_df.attrs["physical_shape"][1],
            pca_h.training_df.attrs["physical_shape"][2],
        )

        # Generate forecast in the initial dimension and reshape.
        forecast_posterior = pca_h.custom_inverse_transform(h_pca_reverse).reshape(
            (self.n_posts, osx, osy)
        )

        return forecast_posterior
