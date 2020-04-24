#  Copyright (c) 2020. Robin Thibaut, Ghent University

import multiprocessing as mp
import os
import shutil
import uuid
from os.path import join as jp

import numpy as np

import bel.toolbox.file_ops as fops
from bel.hydro.backtracking.modpath import backtrack
from bel.hydro.flow.modflow import flow
from bel.hydro.transport.mt3d import transport
from bel.hydro.whpa.travelling_particles import tsp


def simulation(folder=None):
    if folder is 0:
        folder = None
    # Directories
    mod_dir = os.getcwd()  # Module directory
    main_dir = os.path.dirname(mod_dir)
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

    results_dir = jp(main_dir, 'hydro', 'results', res_dir)
    grid_dir = jp(main_dir, 'hydro', 'grid')
    # Generates the result directory
    fops.dirmaker(results_dir)

    # Run Flow
    flow_model = flow(exe_name=exe_name_mf, model_ws=results_dir, grid_dir=grid_dir)
    # # Run Transport
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
    pool = mp.Pool(mp.cpu_count()-1)
    pool.map(simulation, np.zeros(300))


if __name__ == "__main__":
    simulation('new_illustration')

