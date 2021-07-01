#  Copyright (c) 2021. Robin Thibaut, Ghent University
"""

In a Bayesian framework: the prior consists of all possibilities
imagined by a human modeler, possibly aided by computers, then, the posterior includes
only those possibilities that cannot be falsified with data as modeled in the likelihood.
- Modeling Uncertainty in the Earth Sciences, p. 49

If simpler models can be run more frequently
when uncertainty is a critical objective (such as in forecasts), then simpler models may
be preferred if the difference between a simple model and a complex model is small
compared to all other uncertainties in the model.
- Modeling Uncertainty in the Earth Sciences, p. 52

"""

import multiprocessing as mp
import os
import time

import numpy as np
from loguru import logger

import bel4ed.utils
from bel4ed.config import Setup
from bel4ed.hydro import forward_modelling


def main(n_sim: int = None):
    """Main function for multiprocessing"""
    # Automatically selects num,ber of worker based on cpu count
    n_cpu = mp.cpu_count() // 2 + 1
    logger.info(f"working on {n_cpu} cpu - good luck")
    pool = mp.Pool(n_cpu)

    # If n_sim arg is left to None, redo all simulations in folders already presents
    if n_sim is None:
        # List directories in forwards folder
        listme = os.listdir(Setup.Directories.hydro_res_dir)
        folders = list(
            filter(
                lambda d: os.path.isdir(
                    os.path.join(Setup.Directories.hydro_res_dir, d)
                ),
                listme,
            )
        )

    # If n_sim set to -1, perform forward modelling on the folder listed in the file roots.dat
    elif n_sim == -1:
        training_roots = bel4ed.utils.data_read(
            os.path.join(Setup.Directories.forecasts_dir, "base", "roots.dat")
        )
        folders = [item for sublist in training_roots for item in sublist]

    # If n_sim is any positive integer, performs the number of selected forward modelling
    else:
        folders = np.zeros(n_sim)

    # Start processes
    pool.map(forward_modelling, folders)


if __name__ == "__main__":
    start = time.time()
    main(None)
    # forward_modelling("structural_test")
    end = time.time()
    logger.info(f"TET (hours) {(end - start) / 60 / 60}")
