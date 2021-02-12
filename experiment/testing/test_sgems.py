#  Copyright (c) 2021. Robin Thibaut, Ghent University

import multiprocessing as mp
import shutil
import time
import uuid
import os
from os.path import join as jp

import numpy as np

import experiment.toolbox.filesio as fops
from experiment.base.inventory import Machine, MySetup
from experiment.calculation.sgsim import sgsim


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
    fops.dirmaker(results_dir)

    # Statistical simulation
    wells_values = np.ones(len(MySetup._wells.combination) + 1) * -9966699
    hk_array, xy_dummy = sgsim(model_ws=results_dir, grid_dir=grid_dir, wells_hk=wells_values)


if __name__ == "__main__":
    start = time.time()
    test_sgems('macos')
    end = time.time()
    print((end - start) / 60)
