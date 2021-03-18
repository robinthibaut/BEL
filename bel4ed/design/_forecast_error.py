#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
from os.path import join as jp
from typing import Type

import joblib
import numpy as np

from loguru import logger
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.pipeline import Pipeline

from .. import utils
from ..config import Setup
from ..utils import Root
from ..learning.bel import BEL
from ..spatial import (
    contours_vertices,
)

__all__ = [
    "UncertaintyQuantification",
    "measure_info_mode",
    "objective_function",
    "compute_metric",
]


class UncertaintyQuantification:
    def __init__(
        self,
        base: Type[Setup],
        seed: int = None,
    ):
        """
        :param base: class: Base object (inventory)
        :param seed: int: Seed
        """

        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None

        self.base = base
        fc = self.base.Focus
        self.x_lim, self.y_lim, self.grf = fc.x_range, fc.y_range, fc.cell_dim

        # Number of CCA components is chosen as the min number of PC
        n_pc_pred, n_pc_targ = (
            base.HyperParameters.n_pc_predictor,
            base.HyperParameters.n_pc_target,
        )

        # Pipeline before CCA
        self.X_pre_processing = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("pca", PCA()),
            ]
        )
        self.Y_pre_processing = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("pca", PCA()),
            ]
        )

        self.cca = CCA(
            n_components=min(n_pc_targ, n_pc_pred), max_iter=500 * 20, tol=1e-6
        )

        self.X_post_processing = Pipeline(
            [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
        )
        self.Y_post_processing = Pipeline(
            [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
        )

        self.bel = BEL(
            X_pre_processing=self.X_pre_processing,
            X_post_processing=self.X_post_processing,
            Y_pre_processing=self.Y_pre_processing,
            Y_post_processing=self.Y_post_processing,
            cca=self.cca,
        )

        self.X_obs = None

        # Sampling
        self.n_posts = self.base.HyperParameters.n_posts
        self.forecast_posterior = None

        # 0 contours of posterior WHPA
        self.vertices = None


def analysis(bel, X_train, X_test, y_train, directory, source_ids):
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
    combinations = [source_ids]
    total = len(X_test)
    for ix, test_root in enumerate(X_test.index):  # For each observation root
        logger.info(f"[{ix + 1}/{total}]-{test_root}")
        # Directory in which to load forecasts
        bel_dir = jp(directory, test_root)

        for ixw, c in enumerate(combinations):  # For each wel combination
            logger.info(
                f"[{ix + 1}/{total}]-{test_root}-{ixw + 1}/{len(combinations)}"
            )

            new_dir = "".join(
                list(map(str, source_ids))
            )  # sub-directory for forecasts
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
            selection = list(map(str, [wc for wc in source_ids]))
            X_train = X_train.loc[:, selection]
            X_test = X_test.loc[:, selection]

            # BEL fit
            bel.fit(X=X_train, Y=y_train)
            joblib.dump(bel, jp(obj_dir, "bel.pkl"))  # Save the fitted CCA operator
            msg = f"model trained and saved in {obj_dir}"
            logger.info(msg)

            # %% Sample
            # Extract n random sample (target pc's).
            # The posterior distribution is computed within the method below.
            Y_posts_gaussian = bel.predict(X_test)

            Y_posterior = bel.inverse_transform(
                Y_pred=Y_posts_gaussian.reshape(1, -1),
            )

            np.save(jp(obj_dir, "post.npy"), Y_posterior)

            return bel.posterior_mean, bel.posterior_covariance


def objective_function(uq: UncertaintyQuantification, metric):
    """
    Computes the metric between the true WHPA that has been recovered from its n first PCA
    components to allow proper comparison.
    """
    # The idea is to compute the metric with the observed WHPA recovered from it's n first PC.
    n_cut = uq.h_pco.n_pc_cut  # Number of components to keep
    # Inverse fit_transform and reshape
    # FIXME: Problem is that uq.h_pco_predict_pc is None
    true_image = uq.h_pco.custom_inverse_transform(uq.h_pco.predict_pc, n_cut).reshape(
        (uq.shape[1], uq.shape[2])
    )

    method_name = metric.__name__
    logger.info(f"Quantifying image difference based on {method_name}")
    if method_name == "modified_hausdorff":
        x, y = uq.contour_extract()
        to_compare = uq.vertices
        true_feature = contours_vertices(x=x, y=y, arrays=true_image)[0]
    elif method_name == "structural_similarity":
        to_compare = uq.forecast_posterior
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
    base: Type[Setup], roots_obs: Root, combinations: list, metric, base_dir: str = None
):
    if base_dir is None:
        base_dir = base.Directories.forecasts_base_dir

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
