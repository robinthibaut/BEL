import os
import shutil
import uuid
from os.path import join as jp

import numpy as np
from multiprocessing import Process

from bel.hydro.backtracking.modpath import backtrack
from bel.hydro.flow.modflow import flow
from bel.hydro.transport.mt3d import transport
from bel.hydro.whpa.travelling_particles import tsp
from bel.toolbox.file_ops import FileOps

# Directories
mod_dir = os.getcwd()  # Module directory
exe_loc = jp(mod_dir, 'exe')  # EXE directory
# EXE files directory.
exe_name_mf = jp(exe_loc, 'mf2005.exe')
exe_name_mt = jp(exe_loc, 'mt3d.exe')
exe_name_mp = jp(exe_loc, 'mp7.exe')


def main(folder=None):
    if not folder:
        # Main results directory.
        res_dir = uuid.uuid4().hex
    else:
        res_dir = folder

    results_dir = jp(mod_dir, 'results', res_dir)
    # Generates the result directory
    FileOps.dirmaker(results_dir)
    # Loads well information
    wells_data = np.load(jp(mod_dir, 'grid', 'iw'))
    # Run Flow
    flow_model = flow(exe_name=exe_name_mf, model_ws=results_dir, wells=wells_data)
    # # Run Transport
    if flow_model:  # If flow simulation succeeds
        transport(modflowmodel=flow_model, exe_name=exe_name_mt)
        # Run Modpath
        end_points = backtrack(flow_model, exe_name_mp)
        # Compute particle delineation to compute signed distance later on
        delineation = tsp(end_points)  # indices of the vertices of the final protection zone using TSP algorithm
        pzs = end_points[delineation]  # x-y coordinates protection zone
        np.save(jp(results_dir, 'pz'), pzs)  # Save those
        # Deletes everything except final results
        if not folder:
            FileOps.keep_essential(results_dir)
    else:
        shutil.rmtree(results_dir)


if __name__ == "__main__":
    jobs = []
    n_series = 250
    n_jobs = 4
    for j in range(n_series):
        for i in range(n_jobs):  # Can run max 4 instances of mt3dms at once on this computer
            process = Process(target=main)
            jobs.append(process)
            process.start()
        process.join()
        process.close()
