#  Copyright (c) 2021. Robin Thibaut, Ghent University
from os.path import join as jp

import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.pipeline import Pipeline

from bel4ed.algorithms import modified_hausdorff, structural_similarity
from bel4ed.config import Setup
from bel4ed.design import measure_info_mode
from bel4ed.datasets import i_am_root, load_dataset
from bel4ed.design import analysis
from bel4ed.learning.bel import BEL


def main_1(metric=None):
    if metric is None:
        metric = modified_hausdorff

    # Get roots used for testing
    training_file = jp(Setup.Directories.storage_dir, "roots.dat")
    test_file = jp(Setup.Directories.storage_dir, "test_roots.dat")
    training_r, test_r = i_am_root(training_file=training_file, test_file=test_file)

    # Load datasets
    X, Y = load_dataset()

    # Source IDs
    wells = np.array([[1, 2, 3, 4, 5, 6], [1], [2], [3], [4], [5], [6]], dtype=object)
    base = Setup
    base.Wells.combination = wells

    # Select roots for testing
    X_train = X.loc[training_r]
    # X_test = X.loc[["818bf1676c424f76b83bd777ae588a1d"]]
    X_test = X.loc[test_r]
    y_train = Y.loc[training_r]
    # y_test = Y.loc[["818bf1676c424f76b83bd777ae588a1d"]]
    y_test = Y.loc[test_r]

    # Set seed
    seed = 123456
    np.random.seed(seed)

    # Pipelines
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
    bel = BEL(
        X_pre_processing=X_pre_processing,
        X_post_processing=X_post_processing,
        Y_pre_processing=Y_pre_processing,
        Y_post_processing=Y_post_processing,
        cca=cca,
    )

    analysis(
        bel=bel,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        directory=base.Directories.forecasts_dir,
        source_ids=wells,
        metric=modified_hausdorff,
    )

    # 3 - Process dissimilarity measure
    measure_info_mode(base=base, roots_obs=test_r, metric=metric)


if __name__ == "__main__":
    main_1(metric=modified_hausdorff)
