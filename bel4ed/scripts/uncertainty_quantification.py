#  Copyright (c) 2021. Robin Thibaut, Ghent University
from os.path import join as jp

import numpy as np
from loguru import logger
from bel4ed.algorithms import modified_hausdorff, structural_similarity

from bel4ed.config import Setup
from bel4ed.design import UncertaintyQuantification, compute_metric, measure_info_mode
from bel4ed.utils import i_am_root


def main_1(metric=None):

    if metric is None:
        metric = modified_hausdorff

    # 0 - Load training dataset ID's
    training_r, test_r = i_am_root()

    wells = [[1], [2], [3], [4], [5], [6]]
    base = Setup
    base.Wells.combination = wells

    for i, tr in enumerate(test_r):
        for w in wells:
            sub_folder = list(map(str, w))[0]
            uq = UncertaintyQuantification(
                base=base,
                base_dir=None,
                study_folder=jp(tr, sub_folder),
                seed=123456,
            )
            # 1 - Fit / Transform
            uq.analysis(
                roots_training=training_r,
                roots_obs=test_r,
                wipe=False,
                flag_base=False,
            )

    base.Wells.combination = wells  # Not optimal

    # 2 - Sample and compute dissimilarity
    compute_metric(base=base, roots_obs=test_r, combinations=wells, metric=metric)

    # 3 - Process dissimilarity measure
    measure_info_mode(base=base, roots_obs=test_r, metric=metric)


if __name__ == "__main__":
    main_1(metric=modified_hausdorff)
