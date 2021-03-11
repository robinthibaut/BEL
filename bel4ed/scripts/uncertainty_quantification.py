#  Copyright (c) 2021. Robin Thibaut, Ghent University

import numpy as np
from loguru import logger
from bel4ed.algorithms import modified_hausdorff, structural_similarity

from bel4ed.config import Setup
from bel4ed.design import compute_metric, measure_info_mode
from bel4ed.utils import get_roots


def main_1(metric=None):

    if metric is None:
        metric = modified_hausdorff

    # 0 - Load training dataset ID's
    training_r, test_r = get_roots()

    wells = [[1], [2], [3], [4], [5], [6]]
    # wells = [[1, 2, 3, 4, 5, 6]]
    base = Setup
    base.Wells.combination = wells

    # 1 - Fit / Transform
    analysis(
        base=base,
        roots_training=training_r,
        roots_obs=test_r,
        wipe=True,
        flag_base=True,
    )
    base.Wells.combination = wells

    # 2 - Sample and compute dissimilarity
    compute_metric(base=base, roots_obs=test_r, combinations=wells, metric=metric)

    # 3 - Process dissimilarity measure
    measure_info_mode(base=base, roots_obs=test_r, metric=metric)

    # 4 - Plot


def main_2(N, metric=None):
    if metric is None:
        metric = modified_hausdorff
    means = []
    wells = [[1, 2, 3, 4, 5, 6]]
    base = Setup
    base.Wells.combination = wells
    base.ED.metric = metric
    for n in N:
        Setup.HyperParameters.n_total = n
        Setup.HyperParameters.n_training = int(n * 0.8)
        logger.info(f"n_training={int(n * .8)}")
        Setup.HyperParameters.n_test = int(n * 0.2)
        logger.info(f"n_test={int(n * .2)}")

        *_, mhd_mean = analysis(
            base=base,
            n_training=Setup.HyperParameters.n_training,
            n_obs=Setup.HyperParameters.n_test,
            wipe=True,
            flag_base=True,
        )
        means.append([n, mhd_mean])

    return np.array(means)


if __name__ == "__main__":
    main_1(metric=modified_hausdorff)
    # main_1(metric=structural_similarity)
    # n_try = np.linspace(100, 2000, 50)
    # n_try = [100]
    # mv = main_2(N=n_try)
    # np.save(os.path.join(Setup.Directories.storage_dir, "means.npy"), mv)
