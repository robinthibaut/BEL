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
import copy
import multiprocessing as mp
import os
import time

from loguru import logger

import bel4ed.utils
from bel4ed.config import Setup
from bel4ed.hydro import forward_modelling


def main(**kwargs):
    """Main function for multiprocessing"""
    # Automatically selects number of worker based on cpu count
    n_cpu = mp.cpu_count() // 2 + 1
    logger.info(f"working on {n_cpu} cpu - good luck")
    pool = mp.Pool(n_cpu)

    n_sim = kwargs["n_sim"]

    # If n_sim arg is left to None, redo all simulations in folders already presents
    if n_sim is None:
        # List directories in forwards folder
        listme = os.listdir(Setup.Directories.hydro_res_dir)
        folders = list(
            filter(
                lambda fd: os.path.isdir(
                    os.path.join(Setup.Directories.hydro_res_dir, fd)
                ),
                listme,
            )
        )

        if not kwargs["pool"]:
            fwd_dict = kwargs
            for f in folders:
                fwd_dict["folder"] = f
                forward_modelling(**fwd_dict)
            return 0
        else:
            dicts = [copy.deepcopy(kwargs) for _ in range(len(folders))]
            for i, d in enumerate(dicts):
                d.update((k, folders[i]) for k, v in d.items() if k == "folder")

    # If n_sim set to -1, perform forward modelling on the folder listed in the file roots.dat
    elif n_sim == -1:
        training_roots = bel4ed.utils.data_read(
            os.path.join(Setup.Directories.forecasts_dir, "base", "roots.dat")
        )
        folders = [item for sublist in training_roots for item in sublist]
        dicts = [copy.deepcopy(kwargs) for _ in range(len(folders))]
        for i, f in enumerate(folders):
            dicts[i]["folder"] = f

    # If n_sim is any positive integer, performs the number of selected forward modelling
    else:
        dicts = [kwargs for _ in range(n_sim)]

    # Start processes
    pool.map(forward_modelling, dicts)
    pool.close()
    pool.join()


if __name__ == "__main__":
    start = time.time()
    # main(None)
    kw_args_single = {
        "folder": "structural_test",
        "override": True,
        "flow": True,
        "transport": False,
        "ucn": False,
        "backtrack": True,
        "flush": True,
    }
    # forward_modelling(**kw_args_single)
    kw_args_pool = {
        "n_sim": None,
        "folder": 0,
        "pool": True,
        "override": True,
        "flow": True,
        "transport": True,
        "ucn": False,
        "backtrack": False,
        "flush": True,
    }
    main(**kw_args_pool)
    end = time.time()
    logger.info(f"TET (hours) {(end - start) / 60 / 60}")
