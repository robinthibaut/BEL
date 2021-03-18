#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
from os.path import join as jp
from typing import Type

import joblib
import numpy as np
from loguru import logger

from .. import utils
from ..config import Setup
from ..spatial import (
    contours_vertices,
)
from ..spatial import contour_extract
from ..utils import Root

__all__ = [
    "analysis",
    "measure_info_mode",
    "objective_function",
    "compute_metric",
]


def analysis(bel, X_train, X_test, y_train, y_test, directory, source_ids):
    """

    :param bel:
    :param X_train:
    :param X_test:
    :param y_train:
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
                utils.dirmaker(f)
                for f in [
                    obj_dir,
                    fig_data_dir,
                    fig_pca_dir,
                    fig_cca_dir,
                    fig_pred_dir,
                ]
            ]

            # %% Select wells:
            selection = list(map(str, [wc for wc in c]))
            X_train_select = X_train.copy().loc[:, selection]
            X_test_select = X_test.copy().loc[test_root, selection].to_numpy().reshape(1, -1)  # Only one sample
            y_test_select = y_test.copy().loc[test_root].to_numpy().reshape(1, -1)
            bel._y_obs = y_test_select
            # BEL fit
            bel.fit(X=X_train_select, Y=y_train)

            # %% Sample
            # Extract n random sample (target pc's).
            # The posterior distribution is computed within the method below.
            bel.predict(X_test_select)
            Y_posts_gaussian = bel.random_sample()

            Y_posterior = bel.inverse_transform(
                Y_pred=Y_posts_gaussian,
            )

            joblib.dump(bel, jp(obj_dir, "bel.pkl"))  # Save the fitted CCA operator
            msg = f"model trained and saved in {obj_dir}"
            logger.info(msg)
            np.save(jp(obj_dir, "post.npy"), Y_posterior)


def objective_function(bel, metric):
    """
    Computes the metric between the true WHPA that has been recovered from its n first PCA
    components to allow proper comparison.
    """
    # The idea is to compute the metric with the observed WHPA recovered from it's n first PC.
    n_cut = bel.h_pco.n_pc_cut  # Number of components to keep
    # Inverse fit_transform and reshape
    # FIXME: Problem is that bel.h_pco_predict_pc is None
    true_image = bel.h_pco.custom_inverse_transform(
        bel.h_pco.predict_pc, n_cut
    ).reshape((bel.shape[1], bel.shape[2]))

    method_name = metric.__name__
    logger.info(f"Quantifying image difference based on {method_name}")
    if method_name == "modified_hausdorff":
        x, y, vertices = contour_extract()
        to_compare = vertices
        true_feature = contours_vertices(x=x, y=y, arrays=true_image)[0]
    elif method_name == "structural_similarity":
        to_compare = bel.random_sample()
        true_feature = true_image
    else:
        logger.error("Metric name not recognized.")
        to_compare = None
        true_feature = None

    # Compute metric between the 'true image' and the n sampled images or images feature
    similarity = np.array([metric(true_feature, f) for f in to_compare])

    # Save objective_function result
    np.save(jp(uq.res_dir, f"{method_name}"), similarity)

    logger.info(f"Similarity : {np.mean(similarity)}")

    return np.mean(similarity)


def measure_info_mode(base: Type[Setup], roots_obs: Root, metric):
    """Scan the computed metric files and process them based on the mode"""
    logger.info("Computing ED results")

    wid = list(map(str, [_[0] for _ in base.Wells.combination]))  # Well identifiers (n)
    wm = np.zeros((len(wid), base.HyperParameters.n_posts))

    for r in roots_obs:
        # Starting point = root folder in forecast directory
        droot = os.path.join(base.Directories.forecasts_dir, r)
        for e in wid:  # For each sub folder (well) in the main folder
            # Get the objective function file
            ufp = os.path.join(droot, e, "obj")
            fmhd = os.path.join(ufp, f"{metric.__name__}.npy")
            mhd = np.load(fmhd)  # Load MHD
            idw = int(e) - 1  # -1 to respect 0 index (Well index)
            wm[idw] += mhd  # Add MHD at each well

    logger.info("Done")
    np.save(
        os.path.join(base.Directories.forecasts_dir, f"uq_{metric.__name__}.npy"), wm
    )


def compute_metric(
    base: Type[Setup],
    roots_obs: Root,
    combinations: list,
    metric,
):
    global_mean = 0
    total = len(roots_obs)
    for ix, r_ in enumerate(roots_obs):  # For each observation root
        logger.info(f"[{ix + 1}/{total}]-{r_}")
        for ixw, c in enumerate(combinations):  # For each wel combination
            logger.info(f"[{ix + 1}/{total}]-{r_}-{ixw + 1}/{len(combinations)}")
            # Uncertainty analysis
            logger.info("Uncertainty quantification")
            study_folder = os.path.join(
                base.Directories.forecasts_dir, f"{r_}", "".join(list(map(str, c)))
            )
            uq = UncertaintyQuantification(
                base=base,
                seed=123456,
            )
            # Sample posterior
            logger.info("Sample posterior")
            uq.sample_posterior(n_posts=base.HyperParameters.n_posts)
            logger.info("Similarity measure")
            mean = objective_function(uq, metric)
            global_mean += mean

    return global_mean
