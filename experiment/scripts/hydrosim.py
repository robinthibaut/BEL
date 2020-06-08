#  Copyright (c) 2020. Robin Thibaut, Ghent University

import multiprocessing as mp
import shutil
import time
import uuid
from os.path import join as jp

import numpy as np

import experiment.toolbox.filesio as fops
from experiment.base.inventory import Directories
from experiment.hydro.backtracking.modpath import backtrack
from experiment.hydro.flow.modflow import flow
from experiment.hydro.transport.mt3d import transport
from experiment.hydro.whpa.travelling_particles import tsp
from experiment.math.sgsim import sgsim


def simulation(folder=None):
    if folder is 0:
        folder = None
    # Directories
    md = Directories()
    main_dir = md.main_dir
    exe_loc = jp(main_dir, 'hydro', 'exe')  # EXE directory
    # EXE files directory.
    exe_name_mf = jp(exe_loc, 'mf2005.exe')
    exe_name_mt = jp(exe_loc, 'mt3d.exe')
    exe_name_mp = jp(exe_loc, 'mp7.exe')

    if not folder:
        # Main results directory.
        res_dir = uuid.uuid4().hex
    else:
        res_dir = folder

    results_dir = jp(md.hydro_res_dir, res_dir)

    grid_dir = md.grid_dir
    # Generates the result directory
    fops.dirmaker(results_dir)

    # Statistical simulation
    hk_array, xy_dummy = sgsim(model_ws=results_dir, grid_dir=grid_dir)
    # Load previous array:
    hk_array = np.load(jp(md.hydro_res_dir, '46933e56d83d4ddcaa26fa0cd8a795db', 'hk0.npy'))
    # Run Flow
    flow_model = flow(exe_name=exe_name_mf,
                      model_ws=results_dir,
                      grid_dir=grid_dir,
                      hk_array=hk_array, xy_dummy=xy_dummy)
    # Run Transport
    if flow_model:  # If flow simulation succeeds
        transport(modflowmodel=flow_model, exe_name=exe_name_mt, grid_dir=grid_dir)
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
    pool = mp.Pool(mp.cpu_count() - 1)
    pool.map(simulation, np.zeros(300))


if __name__ == "__main__":
    start = time.time()
    simulation('example')
    end = time.time()
    print((end - start) / 60)
