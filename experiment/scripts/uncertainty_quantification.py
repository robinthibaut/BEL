#  Copyright (c) 2021. Robin Thibaut, Ghent University

import numpy as np

from experiment.config import Setup
from experiment.design.forecast_error import analysis
from experiment.testing._test import _test
from experiment.utils import get_roots


def main_1():
    training_r, test_r = get_roots()

    # wells = [[1, 2, 3, 4, 5, 6], [1], [2], [3], [4], [5], [6]]
    wells = [[1, 2, 3, 4, 5, 6]]
    analysis(
        base=Setup,
        comb=wells,
        roots_training=training_r,
        roots_obs=test_r,
        wipe=False,
        flag_base=True,
    )


def main_2(N):
    means = []

    for n in N:
        Setup.HyperParameters.n_total = n
        Setup.HyperParameters.n_training = int(n * 0.8)
        print(f"n_training={int(n * .8)}")
        Setup.HyperParameters.n_test = int(n * 0.2)
        print(f"n_test={int(n * .2)}")

        # wells = [[1, 2, 3, 4, 5, 6], [1], [2], [3], [4], [5], [6]]
        wells = [[1, 2, 3, 4, 5, 6]]
        *_, mhd_mean = analysis(
            base=Setup,
            comb=wells,
            n_training=Setup.HyperParameters.n_training,
            n_obs=Setup.HyperParameters.n_test,
            wipe=True,
            flag_base=True,
        )
        means.append([n, mhd_mean])

    return np.array(means)


if __name__ == "__main__":
    _test()
    # main_1()
    # n_try = np.linspace(100, 2000, 50)
    # n_try = [100]
    # mv = main_2(N=n_try)
    # np.save(os.path.join(Setup.Directories.storage_dir, "means.npy"), mv)
