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
import shutil
import time
import uuid
from os.path import join as jp

import numpy as np

import experiment.utils
from experiment.algorithms.statistics import sgsim
from experiment.config import Machine, Setup
from experiment.hydro.backtracking.modpath import backtrack
from experiment.hydro.flow.modflow import flow
from experiment.hydro.transport.mt3d import transport
from experiment.processing.target_handle import travelling_particles


def simulation(folder=None):
    """Data collection"""
    if folder == 0:
        folder = None
    # Directories
    main_dir = Setup.Directories.main_dir
    exe_loc = jp(main_dir, "hydro", "exe")
    # EXE files directory.
    if Machine.computer == "MacBook-Pro.local":
        exe_name_mf = jp(exe_loc, "mf2005")
        exe_name_mt = jp(exe_loc, "mt3dms")
        exe_name_mp = jp(exe_loc, "mp7")
    else:
        exe_name_mf = jp(exe_loc, "mf2005.exe")
        exe_name_mt = jp(exe_loc, "mt3d.exe")
        exe_name_mp = jp(exe_loc, "mp7.exe")

    if not folder:
        # Main results directory.
        res_dir = uuid.uuid4().hex
    else:
        res_dir = folder

    results_dir = jp(Setup.Directories.hydro_res_dir, res_dir)

    grid_dir = Setup.Directories.grid_dir
    # Generates the result directory
    experiment.utils.dirmaker(results_dir)

    print(f"fwd {res_dir}")
    # Check if forwards have already been computed
    opt = np.array(
        [os.path.isfile(jp(results_dir, d)) for d in Setup.Files.output_files])

    if not opt.all():
        # Resets folder
        # fops.folder_reset(results_dir, exceptions=MySetup.Files.sgems_family)

        start_fwd = time.time()
        # Statistical simulation
        hk_array, xy_dummy = sgsim(model_ws=results_dir, grid_dir=grid_dir)

        # Run Flow
        flow_model = flow(
            exe_name=exe_name_mf,
            model_ws=results_dir,
            grid_dir=grid_dir,
            hk_array=hk_array,
            xy_dummy=xy_dummy,
        )
        # Run Transport and Backtracking
        if flow_model:  # If flow simulation succeeds
            transport(
                modflowmodel=flow_model,
                exe_name=exe_name_mt,
                grid_dir=grid_dir,
                save_ucn=False,
            )
            # Run Modpath
            end_points = backtrack(flow_model, exe_name_mp)
            # Compute particle delineation to compute signed distance later on
            # indices of the vertices of the final protection zone
            delineation = travelling_particles(end_points)
            # using TSP algorithm
            pzs = end_points[delineation]  # x-y coordinates protection zone
            np.save(jp(results_dir, "pz"), pzs)  # Save those
            # Deletes everything except final results
            hl = (time.time() - start_fwd) // 60
            print(f"done in {hl} min")
            if not folder:
                experiment.utils.keep_essential(results_dir)
        else:
            shutil.rmtree(results_dir)
            print(f"terminated f{res_dir}")
            return 0
    else:
        print(f"pass {res_dir}")
        hk_array, xy_dummy = sgsim(model_ws=results_dir, grid_dir=grid_dir)
        # Run Flow Modelling
        flow(
            exe_name=exe_name_mf,
            model_ws=results_dir,
            grid_dir=grid_dir,
            hk_array=hk_array,
            xy_dummy=xy_dummy,
        )

    return results_dir


def main(n_sim: int = None):
    """Main function for multiprocessing"""
    # Automatically selects num,ber of worker based on cpu count
    n_cpu = mp.cpu_count() // 2 + 1
    print(f"working on {n_cpu} cpu - good luck")
    pool = mp.Pool(n_cpu)

    # If n_sim arg is left to None, redo all simulations in folders already presents
    if n_sim is None:
        # List directories in forwards folder
        listme = os.listdir(Setup.Directories.hydro_res_dir)
        folders = list(
            filter(
                lambda d: os.path.isdir(
                    os.path.join(Setup.Directories.hydro_res_dir, d)),
                listme,
            ))

    # If n_sim set to -1, perform forward modelling on the folder listed in the file roots.dat
    elif n_sim == -1:
        training_roots = experiment.utils.data_read(
            os.path.join(Setup.Directories.forecasts_dir, "base", "roots.dat"))
        folders = [item for sublist in training_roots for item in sublist]

    # If n_sim is any positive integer, performs the number of selected forward modelling
    else:
        folders = np.zeros(n_sim)

    # Start processes
    pool.map(simulation, folders)


def test():
    name = "test"
    return simulation(name)


if __name__ == "__main__":
    start = time.time()
    simulation("test190221")
    # main(10)
    end = time.time()
    print(f"TET (hours) {(end - start) / 60 / 60}")
