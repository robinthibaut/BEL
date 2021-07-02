#  Copyright (c) 2021. Robin Thibaut, Ghent University
import os
import sys

from datetime import date
from os.path import join as jp

from loguru import logger
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

try:
    sys.path.insert(0, jp("..", "..", "..", "PycharmProjects", "pysgems"))
    sys.path.insert(0, jp("..", "..", "..", "PycharmProjects", "skbel"))
except Exception as e:
    logger.warning(e)

from bel4ed.config import Setup
from bel4ed.hydro import forward_modelling

from skbel.learning.bel import BEL

__version__ = "1.0.dev0"

source = __name__.split(".")[-1]
# Set up logger
logger.add(
    jp(os.getcwd(), "logs", f"{source}_{date.today()}.log"),
    backtrace=True,
    diagnose=True,
    enqueue=True,
)
logger.debug(f"Beginning logging session for {source}!")

__all__ = [
    "config",
    "utils",
    "goggles",
    "design",
    "hydro",
    "preprocessing",
    "spatial",
    "init_bel",
]


def kernel_bel():
    """
    Set all BEL pipelines. This is the blueprint of the framework.
    """
    n_pc_pred, n_pc_targ = 80, 100
    # Pipeline before CCA
    X_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                KernelPCA(
                    n_components=n_pc_pred,
                    kernel="rbf",
                    fit_inverse_transform=False,
                    alpha=1e-5,
                ),
            ),
        ]
    )
    Y_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                KernelPCA(
                    n_components=n_pc_targ,
                    kernel="rbf",
                    fit_inverse_transform=True,
                    alpha=1e-5,
                ),
            ),
        ]
    )

    # Canonical Correlation Analysis
    cca = CCA(n_components=min(n_pc_targ, n_pc_pred), max_iter=500 * 5)

    # Pipeline after CCA
    X_post_processing = Pipeline(
        [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
    )
    Y_post_processing = Pipeline(
        [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
    )

    # Initiate BEL object
    bel_model = BEL(
        X_pre_processing=X_pre_processing,
        X_post_processing=X_post_processing,
        Y_pre_processing=Y_pre_processing,
        Y_post_processing=Y_post_processing,
        cca=cca,
    )

    bel_model.X_n_pc = n_pc_pred
    bel_model.Y_n_pc = n_pc_targ

    return bel_model


def init_bel():
    """
    Set all BEL pipelines
    :return:
    """
    n_pc_pred, n_pc_targ = (
        Setup.HyperParameters.n_pc_predictor,
        Setup.HyperParameters.n_pc_target,
    )
    # Pipeline before CCA
    X_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_pc_pred)),
        ]
    )
    Y_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_pc_targ)),
        ]
    )

    # Canonical Correlation Analysis
    # Number of CCA components is chosen as the min number of PC

    cca = CCA(n_components=min(n_pc_targ, n_pc_pred), max_iter=500 * 5)

    # Pipeline after CCA
    X_post_processing = Pipeline(
        [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
    )
    Y_post_processing = Pipeline(
        [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
    )

    # Initiate BEL object
    bel_model = BEL(
        X_pre_processing=X_pre_processing,
        X_post_processing=X_post_processing,
        Y_pre_processing=Y_pre_processing,
        Y_post_processing=Y_post_processing,
        cca=cca,
    )

    # Set PC cut
    bel_model.X_n_pc = n_pc_pred
    bel_model.Y_n_pc = n_pc_targ

    return bel_model
