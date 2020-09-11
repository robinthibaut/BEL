#  Copyright (c) 2020. Robin Thibaut, Ghent University

"""

In a Bayesian framework: the prior consists of all possibilities
imagined by a human modeler, possibly aided by computers, then, the posterior includes
only those possibilities that cannot be falsified with data as modeled in the likelihood.
- Jef Caers, Modeling Uncertainty in the Earth Sciences, p. 49

If simpler models can be run more frequently
when uncertainty is a critical objective (such as in forecasts), then simpler models may
be preferred if the difference between a simple model and a complex model is small
compared to all other uncertainties in the model.
- Jef Caers, Modeling Uncertainty in the Earth Sciences, p. 52

"""


import multiprocessing as mp
import shutil
import time
import uuid
import os
from os.path import join as jp

import numpy as np

import experiment.toolbox.filesio as fops
from experiment.base.inventory import Machine, MySetup
from experiment.hydro.backtracking.modpath import backtrack
from experiment.hydro.flow.modflow import flow
from experiment.hydro.transport.mt3d import transport
from experiment.hydro.whpa.travelling_particles import tsp
from experiment.math.sgsim import sgsim


def simulation(folder=None):
    if folder == 0:
        folder = None
    # Directories
    main_dir = MySetup.Directories.main_dir
    exe_loc = jp(main_dir, 'hydro', 'exe')  #
    # directory
    # EXE files directory.
    if Machine.computer == 'MacBook-Pro.local':
        exe_name_mf = jp(exe_loc, 'mf2005')
        exe_name_mt = jp(exe_loc, 'mt3dms')
        exe_name_mp = jp(exe_loc, 'mp7')
    else:
        exe_name_mf = jp(exe_loc, 'mf2005.exe')
        exe_name_mt = jp(exe_loc, 'mt3d.exe')
        exe_name_mp = jp(exe_loc, 'mp7.exe')

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
    hk_array, xy_dummy = sgsim(model_ws=results_dir, grid_dir=grid_dir)

    # Run Flow
    flow_model = flow(exe_name=exe_name_mf,
                      model_ws=results_dir,
                      grid_dir=grid_dir,
                      hk_array=hk_array, xy_dummy=xy_dummy)
    # Run Transport
    if flow_model:  # If flow simulation succeeds
        transport(modflowmodel=flow_model, exe_name=exe_name_mt, grid_dir=grid_dir, save_ucn=True)
        # Run Modpath
        end_points = backtrack(flow_model, exe_name_mp)
        # Compute particle delineation to compute signed distance later on
        delineation = tsp(end_points)  # indices of the vertices of the final protection zone using TSP algorithm
        pzs = end_points[delineation]  # x-y coordinates protection zone
        np.save(jp(results_dir, 'pz'), pzs)  # Save those
        # Deletes everything except final results
        if not folder:
            fops.keep_essential(results_dir)
    else:
        shutil.rmtree(results_dir)


def main():
    n_cpu = mp.cpu_count()//2 + 1
    pool = mp.Pool(n_cpu)
    pool.map(simulation, np.zeros(100))


if __name__ == "__main__":
    start = time.time()
    simulation('46d0170062654fc3b36888f2e2510fcb')
    # main()
    end = time.time()
    print((end - start) / 60)
