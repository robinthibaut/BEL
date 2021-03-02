#  Copyright (c) 2021. Robin Thibaut, Ghent University

import time
import uuid
from os.path import join as jp

import numpy as np

import experiment._utils as fops
import experiment.utils
from experiment.algorithms.statistics import sgsim
from experiment.algorithms.base import MySetup


def test_sgems(folder=None):
    if folder is 0:
        folder = None

    if not folder:
        # Main results directory.
        res_dir = uuid.uuid4().hex
    else:
        res_dir = folder

    results_dir = jp(MySetup.Directories.hydro_res_dir, res_dir)

    grid_dir = MySetup.Directories.grid_dir
    # Generates the result directory
    experiment.utils.dirmaker(results_dir)

    # Statistical simulation
    wells_values = np.ones(len(MySetup.Wells.combination) + 1) * -9966699
    hk_array, xy_dummy = sgsim(
        model_ws=results_dir, grid_dir=grid_dir, wells_hk=wells_values)


if __name__ == "__main__":
    start = time.time()
    test_sgems('macos')
    end = time.time()
    print((end - start) / 60)
