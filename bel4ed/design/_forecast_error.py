#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import joblib
import numpy as np
from loguru import logger
from sklearn.base import clone

from .. import utils
from ..config import Setup
from ..spatial import (
    contours_vertices,
)
from ..spatial import contour_extract
from ..utils import Root

__all__ = [
    "bel_training",
    "bel_uq",
    "measure_info_mode",
]


def bel_training(bel, X_train, X_test, y_train, y_test, directory, source_ids):
    """
    :param bel:
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param directory:
    :param source_ids:
    :return:
    """
    # Directories
    combinations = source_ids
    total = len(X_test)
    for ix, test_root in enumerate(X_test.index):  # For each observation root
        logger.info(f"[{ix + 1}/{total}]-{test_root}")
        # Directory in which to load forecasts
        bel_dir = jp(directory, test_root)

        for ixw, c in enumerate(combinations):  # For each wel combination
            logger.info(f"[{ix + 1}/{total}]-{test_root}-{ixw + 1}/{len(combinations)}")

            new_dir = "".join(list(map(str, c)))  # sub-directory for forecasts
            sub_dir = jp(bel_dir, new_dir)

            # %% Folders
            obj_dir = jp(sub_dir, "obj")
            fig_data_dir = jp(sub_dir, "data")
            fig_pca_dir = jp(sub_dir, "pca")
            fig_cca_dir = jp(sub_dir, "cca")
            fig_pred_dir = jp(sub_dir, "uq")

            # %% Creates directories
            [
                utils.dirmaker(f, erase=True)
                for f in [
                    obj_dir,
                    fig_data_dir,
                    fig_pca_dir,
                    fig_cca_dir,
                    fig_pred_dir,
                ]
            ]
            # Clone BEL for safety
            bel_clone = clone(bel)
            bel_clone.X_n_pc = Setup.HyperParameters.n_pc_predictor
            bel_clone.Y_n_pc = Setup.HyperParameters.n_pc_target
            bel_clone.n_posts = Setup.HyperParameters.n_posts
            bel_clone.seed = 123456
            # %% Select wells:
            selection = list(map(str, [wc for wc in c]))
            X_train_select = X_train.copy().loc[:, selection]
            # Update physical shape
            X_train_select.attrs["physical_shape"] = (
                len(selection),
                X_train.attrs["physical_shape"][1],
            )
            X_test_select = (
                X_test.copy().loc[test_root, selection].to_numpy().reshape(1, -1)
            )  # Only one sample
            y_test_select = y_test.copy().loc[test_root].to_numpy().reshape(1, -1)
            bel_clone.Y_obs = y_test_select
            # BEL fit
            bel_clone.fit(X=X_train_select, Y=y_train)

            # %% Sample
            # Extract n random sample (target pc's).
            # The posterior distribution is computed within the method below.
            bel_clone.predict(X_test_select)

            # Save the fitted BEL model
            joblib.dump(bel_clone, jp(obj_dir, "bel.pkl"))
            msg = f"model trained and saved in {obj_dir}"
            logger.info(msg)


def bel_uq(
    index: list, directory: str, source_ids: list or np.array, metrics: list or tuple
):
    # Directories
    combinations = source_ids
    total = len(index)
    for ix, test_root in enumerate(index):  # For each observation root
        logger.info(f"[{ix + 1}/{total}]-{test_root}")
        # Directory in which to load forecasts
        bel_dir = jp(directory, test_root)

        for ixw, c in enumerate(combinations):  # For each wel combination
            logger.info(f"[{ix + 1}/{total}]-{test_root}-{ixw + 1}/{len(combinations)}")

            new_dir = "".join(list(map(str, c)))  # sub-directory for forecasts
            sub_dir = jp(bel_dir, new_dir)

            # %% Folders
            obj_dir = jp(sub_dir, "obj")
            bel = joblib.load(jp(obj_dir, "bel.pkl"))
            # Compute objective function
            # The idea is to compute the metric with the observed WHPA recovered from it's n first PC.
            n_cut = bel.Y_n_pc  # Number of components to keep
            y_obs_pc = bel.Y_pre_processing.transform(bel.Y_obs)
            dummy = np.zeros(
                (1, y_obs_pc.shape[1])
            )  # Create a dummy matrix filled with zeros
            dummy[:, :n_cut] = y_obs_pc[
                :, :n_cut
            ]  # Fill the dummy matrix with the posterior PC

            # Reshape for the objective function
            Y_reconstructed = bel.Y_pre_processing.inverse_transform(dummy).reshape(
                bel.Y_shape
            )  # Inverse transform = "True image"

            # Compute CCA Gaussian scores
            Y_posts_gaussian = bel.random_sample(n_posts=Setup.HyperParameters.n_posts)
            # Get back to original space
            Y_posterior = bel.inverse_transform(
                Y_pred=Y_posts_gaussian,
            )
            Y_posterior = Y_posterior.reshape(
                (bel.n_posts,) + (bel.Y_shape[1], bel.Y_shape[2])
            )

            for m in metrics:
                _objective_function(
                    y_r=Y_reconstructed,
                    y_samples=Y_posterior,
                    metric=m,
                    directory=obj_dir,
                )


def _objective_function(y_r, y_samples, metric, directory):
    """
    Computes the metric between the true WHPA that has been recovered from its n first PCA
    components to allow proper comparison.
    """

    x_lim, y_lim, grf = Setup.Focus.x_range, Setup.Focus.y_range, Setup.Focus.cell_dim

    method_name = metric.__name__
    logger.info(f"Quantifying image difference based on {method_name}")
    if method_name == "modified_hausdorff":
        # For MHD, we need the 0 contours vertices
        x, y, vertices = contour_extract(x_lim=x_lim, y_lim=y_lim, grf=grf, Z=y_r)
        true_feature = vertices[0]  # Vertices of the true observation
        to_compare = contours_vertices(x=x, y=y, arrays=y_samples)
        similarity = np.array([metric(true_feature, f) for f in to_compare])
    elif method_name == "structural_similarity":
        # SSIM works with continuous images.
        # With SSIM, a value of 1 = perfect similarity.
        # SSIM values decrease with dissimilarity
        true_feature = y_r[0]
        to_compare = y_samples
        # Compute metric between the 'true image' and the n sampled images or images feature
        similarity = 1 - np.array([metric(true_feature, f) for f in to_compare])
    else:
        logger.error("Metric name not recognized.")
        similarity = None

    # Save _objective_function result
    np.save(jp(directory, f"{method_name}"), similarity)

    logger.info(f"Similarity : {np.mean(similarity)}")

    return np.mean(similarity)


def measure_info_mode(roots_obs: Root, metric, source_ids):
    """Scan the computed metric files and process them based on the mode"""
    logger.info("Computing ED results")

    wid = list(map(str, [_[0] for _ in source_ids]))  # Well identifiers (n)
    wm = np.zeros((len(wid), Setup.HyperParameters.n_posts))

    for r in roots_obs:
        # Starting point = root folder in forecast directory
        droot = os.path.join(Setup.Directories.forecasts_dir, r)
        for e in wid:  # For each sub folder (well) in the main folder
            # Get the objective function file
            ufp = os.path.join(droot, e, "obj")
            fmhd = os.path.join(ufp, f"{metric.__name__}.npy")
            mhd = np.load(fmhd)  # Load MHD
            idw = int(e) - 1  # -1 to respect 0 index (Well index)
            wm[idw] += mhd  # Add MHD at each well

    logger.info("Done")
    np.save(
        os.path.join(Setup.Directories.forecasts_dir, f"uq_{metric.__name__}.npy"), wm
    )
