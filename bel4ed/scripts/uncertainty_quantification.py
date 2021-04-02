#  Copyright (c) 2021. Robin Thibaut, Ghent University
from os.path import join as jp

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

from bel4ed.algorithms import modified_hausdorff, structural_similarity
from bel4ed.config import Setup
from bel4ed.datasets import i_am_root, load_dataset
from bel4ed.design import bel_training, bel_uq
from bel4ed.design import measure_info_mode
from bel4ed.goggles import mode_histo
from bel4ed.learning.bel import BEL


def init_bel():
    """
    Set all BEL pipelines
    :return:
    """
    # Pipeline before CCA
    X_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("pca", PCA()),
        ]
    )
    Y_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("pca", PCA()),
        ]
    )

    # Canonical Correlation Analysis
    # Number of CCA components is chosen as the min number of PC
    n_pc_pred, n_pc_targ = (
        Setup.HyperParameters.n_pc_predictor,
        Setup.HyperParameters.n_pc_target,
    )
    cca = CCA(n_components=min(n_pc_targ, n_pc_pred), max_iter=500 * 20, tol=1e-6)

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


def train(model, training_idx: list, test_idx: list, source_ids: list or np.array):
    # Load datasets
    X, Y = load_dataset()

    # Select roots for testing
    X_train = X.loc[training_idx]
    # X_test = X.loc[["818bf1676c424f76b83bd777ae588a1d"]]
    X_test = X.loc[test_idx]
    y_train = Y.loc[training_idx]
    # y_test = Y.loc[["818bf1676c424f76b83bd777ae588a1d"]]
    y_test = Y.loc[test_idx]

    bel_training(
        bel=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        directory=Setup.Directories.forecasts_dir,
        source_ids=source_ids,
    )


def plot_uq(metric_function):
    wm = np.load(
        jp(Setup.Directories.forecasts_dir, f"uq_{metric_function.__name__}.npy")
    )
    colors = Setup.Wells.colors
    mode_histo(colors=colors, wm=wm, an_i=0, fig_name=metric_function.__name__)


if __name__ == "__main__":
    # Get roots used for testing
    training_file = jp(Setup.Directories.storage_dir, "roots.dat")
    test_file = jp(Setup.Directories.storage_dir, "test_roots.dat")
    training_r, test_r = i_am_root(training_file=training_file, test_file=test_file)

    # Source IDs
    wells = np.array([[1], [2], [3], [4], [5], [6]], dtype=object)
    # wells = np.array([[1, 2, 3, 4, 5, 6]], dtype=object)

    # Initiate BEL model
    # bel = init_bel()

    # Train model
    # train(model=bel, training_idx=training_r, test_idx=test_r, source_ids=wells)

    # Pick metrics
    metrics = (modified_hausdorff, structural_similarity)

    # Compute UQ with metrics
    bel_uq(
        index=test_r,
        directory=Setup.Directories.forecasts_dir,
        source_ids=wells,
        metrics=metrics,
    )

    # Process dissimilarity measure
    for m in metrics:
        measure_info_mode(roots_obs=test_r, metric=m, source_ids=wells)

    # Plot UQ
    plot_uq(modified_hausdorff)
    plot_uq(structural_similarity)
