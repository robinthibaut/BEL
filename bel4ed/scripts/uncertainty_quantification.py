#  Copyright (c) 2021. Robin Thibaut, Ghent University

import numpy as np
from loguru import logger
from bel4ed.algorithms import modified_hausdorff, structural_similarity

from bel4ed.config import Setup
from bel4ed.design import analysis
from bel4ed.utils import get_roots


def main_1(metric=None):
    if metric is None:
        metric = modified_hausdorff
    training_r, test_r = get_roots()

    wells = [[1], [2], [3], [4], [5], [6]]
    # wells = [[1, 2, 3, 4, 5, 6]]
    base = Setup
    base.Wells.combination = wells
    analysis(
        base=Setup,
        roots_training=training_r,
        metric=metric,
        roots_obs=test_r,
        wipe=False,
        flag_base=True,
    )


def main_2(N, metric=None):
    if metric is None:
        metric = modified_hausdorff
    means = []
    wells = [[1, 2, 3, 4, 5, 6]]
    base = Setup
    base.Wells.combination = wells
    for n in N:
        Setup.HyperParameters.n_total = n
        Setup.HyperParameters.n_training = int(n * 0.8)
        logger.info(f"n_training={int(n * .8)}")
        Setup.HyperParameters.n_test = int(n * 0.2)
        logger.info(f"n_test={int(n * .2)}")

        *_, mhd_mean = analysis(
            base=base,
            metric=metric,
            n_training=Setup.HyperParameters.n_training,
            n_obs=Setup.HyperParameters.n_test,
            wipe=True,
            flag_base=True,
        )
        means.append([n, mhd_mean])

    return np.array(means)


if __name__ == "__main__":
    main_1()
    # n_try = np.linspace(100, 2000, 50)
    # n_try = [100]
    # mv = main_2(N=n_try)
    # np.save(os.path.join(Setup.Directories.storage_dir, "means.npy"), mv)
