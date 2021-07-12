#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
from copy import deepcopy
from os.path import join as jp, isfile

import joblib
import numpy as np
from loguru import logger
from skbel.spatial import contour_extract
from skbel.spatial import (
    contours_vertices,
)

from .. import utils, init_bel
from ..config import Setup

__all__ = [
    "bel_training",
    "bel_training_mp",
    "bel_uq",
    "bel_uq_mp",
]


def bel_training(bel, *, X_train, X_test, y_train, y_test, directory, source_ids):
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

        for ixw, c in enumerate(combinations):  # For each well combination
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
            n_posts = bel.n_posts
            X_n_pc = bel.X_n_pc
            Y_n_pc = bel.Y_n_pc
            # Reset params
            # bel_clone = clone(bel)
            bel_clone = bel
            # bel_clone.X_n_pc = X_n_pc
            # bel_clone.Y_n_pc = Y_n_pc
            # bel_clone.n_posts = n_posts
            # Setting the seed might cause issues
            # bel_clone.seed = 123456
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


def bel_training_mp(args):
    """"""
    bel, X_train, X_test, y_train, y_test, directory, source_ids, test_root = args
    # Directories
    combinations = source_ids
    # Directory in which to load forecasts
    bel_dir = jp(directory, test_root)

    for ixw, c in enumerate(combinations):  # For each well combination
        new_dir = "".join(list(map(str, c)))  # sub-directory for forecasts
        sub_dir = jp(bel_dir, new_dir)
        obj_dir = jp(sub_dir, "obj")

        if not isfile(jp(obj_dir, "bel.pkl")):
            # %% Folders
            fig_data_dir = jp(sub_dir, "data")
            fig_pca_dir = jp(sub_dir, "pca")
            fig_cca_dir = jp(sub_dir, "cca")
            fig_pred_dir = jp(sub_dir, "uq")

            # %% Creates directories
            [
                utils.dirmaker(f, erase=False)
                for f in [
                    obj_dir,
                    fig_data_dir,
                    fig_pca_dir,
                    fig_cca_dir,
                    fig_pred_dir,
                ]
            ]
            bel_clone = bel
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


def bel_uq(
    *,
    bel,
    y_obs: np.array = None,
    index: list,
    directory: str,
    source_ids: list or np.array,
    metrics: list or tuple,
    delete: bool = False,
):
    metrics = list(metrics)
    # Directories
    combinations = source_ids
    total = len(index)
    wid = list(map(str, [_[0] for _ in source_ids]))  # Well identifiers (n)
    theta = np.zeros((len(metrics), len(wid), bel.n_posts))
    for ix, test_root in enumerate(index):  # For each observation root
        logger.info(f"[{ix + 1}/{total}]-{test_root}")
        # Directory in which to load forecasts
        bel_dir = jp(directory, test_root)

        for ixw, c in enumerate(combinations):  # For each wel combination
            logger.info(f"[{ix + 1}/{total}]-{test_root}-{ixw + 1}/{len(combinations)}")

            new_dir = "".join(list(map(str, c)))  # sub-directory for forecasts
            sub_dir = jp(bel_dir, new_dir)

            # Folders
            obj_dir = jp(sub_dir, "obj")
            bel = joblib.load(jp(obj_dir, "bel.pkl"))

            metrics_copy = deepcopy(metrics)
            for k, m in enumerate(metrics):
                efile = os.path.join(obj_dir, f"uq_{m.__name__}.npy")
                if isfile(efile):
                    metrics_copy.remove(m)
                    logger.info(f"skipping {m.__name__}")
                    oe = np.load(efile)
                    theta[k, ixw] += oe

            if len(metrics_copy) > 0:
                # The idea is to compute the metric with the observed WHPA recovered from it's n first PC.
                n_cut = bel.Y_n_pc  # Number of components to keep
                try:
                    y_obs_pc = bel.Y_pre_processing.transform(y_obs[ix])
                except ValueError:
                    y_obs_pc = bel.Y_pre_processing.transform(
                        y_obs.to_numpy()[ix].reshape(1, -1)
                    )
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
                Y_posts_gaussian = bel.random_sample(n_posts=None)
                # Get back to original space
                Y_posterior = bel.inverse_transform(
                    Y_pred=Y_posts_gaussian,
                )
                Y_posterior = Y_posterior.reshape(
                    (bel.n_posts,) + (bel.Y_shape[1], bel.Y_shape[2])
                )

                for j, m in enumerate(metrics):
                    if m in metrics_copy:
                        oe = _objective_function(
                            y_true=Y_reconstructed,
                            y_pred=Y_posterior,
                            metric=m,
                        )
                        theta[j, ixw] += oe
                        np.save(os.path.join(directory, f"uq_{m.__name__}.npy"), oe)
                if delete:
                    os.remove(jp(obj_dir, "bel.pkl"))

    for k, m in enumerate(metrics):
        np.save(os.path.join(directory, f"uq_{m.__name__}.npy"), theta[k])


def bel_uq_mp(args):
    bel, y_obs, test_root, directory, source_ids, metrics, delete = args
    metrics = list(metrics)
    # Directories
    combinations = source_ids
    # Directory in which to load forecasts
    bel_dir = jp(directory, test_root)

    for ixw, c in enumerate(combinations):  # For each well combination
        new_dir = "".join(list(map(str, c)))  # sub-directory for forecasts
        sub_dir = jp(bel_dir, new_dir)

        # Folders
        obj_dir = jp(sub_dir, "obj")
        bel = joblib.load(jp(obj_dir, "bel.pkl"))

        metrics_copy = deepcopy(metrics)
        for k, m in enumerate(metrics):
            efile = os.path.join(obj_dir, f"uq_{m.__name__}.npy")
            if isfile(efile):
                metrics_copy.remove(m)
                logger.info(f"skipping {m.__name__}")

        if len(metrics_copy) > 0:
            # The idea is to compute the metric with the observed WHPA recovered from it's n first PC.
            n_cut = bel.Y_n_pc  # Number of components to keep
            try:
                y_obs_pc = bel.Y_pre_processing.transform(y_obs.loc[test_root])
            except ValueError:
                y_obs_pc = bel.Y_pre_processing.transform(
                    y_obs.loc[test_root].to_numpy().reshape(1, -1)
                )
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
            Y_posts_gaussian = bel.random_sample(n_posts=None)
            # Get back to original space
            Y_posterior = bel.inverse_transform(
                Y_pred=Y_posts_gaussian,
            )
            Y_posterior = Y_posterior.reshape(
                (bel.n_posts,) + (bel.Y_shape[1], bel.Y_shape[2])
            )

            for j, m in enumerate(metrics):
                if m in metrics_copy:
                    oe = _objective_function(
                        y_true=Y_reconstructed,
                        y_pred=Y_posterior,
                        metric=m,
                    )
                    np.save(os.path.join(directory, f"uq_{m.__name__}.npy"), oe)
            if delete:
                os.remove(jp(obj_dir, "bel.pkl"))


def _objective_function(
    y_true, y_pred, metric, multioutput="raw_values", sample_weight=None
):
    """
    Computes the metric between the true WHPA that has been recovered from its n first PCA
    components to allow proper comparison.
    """

    # TODO: pass this as kwargs
    x_lim, y_lim, grf = Setup.Focus.x_range, Setup.Focus.y_range, Setup.Focus.cell_dim

    method_name = metric.__name__
    logger.info(f"Quantifying image difference based on {method_name}")
    if method_name == "modified_hausdorff":
        # For MHD, we need the 0 contours vertices
        x, y, vertices = contour_extract(x_lim=x_lim, y_lim=y_lim, grf=grf, Z=y_true)
        true_feature = vertices[0]  # Vertices of the true observation
        to_compare = contours_vertices(x=x, y=y, arrays=y_pred)
        output_errors = np.array([metric(true_feature, f) for f in to_compare])
    elif method_name == "structural_similarity":
        # SSIM works with continuous images.
        # With SSIM, a value of 1 = perfect similarity.
        # SSIM values decrease with dissimilarity
        true_feature = y_true[0]
        to_compare = y_pred
        # Compute metric between the 'true image' and the n sampled images or images feature
        output_errors = -np.array([metric(true_feature, f) for f in to_compare])
    else:
        logger.error("Metric name not recognized.")
        output_errors = None

    logger.info(f"Similarity : {np.average(output_errors, weights=sample_weight)}")

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None
            return np.average(output_errors, weights=multioutput, axis=0)
