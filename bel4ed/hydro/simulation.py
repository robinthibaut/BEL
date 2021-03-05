#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
import shutil
import time
import uuid
from os.path import join as jp

import numpy as np

import bel4ed.utils
from bel4ed.algorithms.statistics import sgsim
from bel4ed.config import Setup, Machine
from bel4ed.hydro.backtracking.modpath import backtrack
from bel4ed.hydro.flow.modflow import flow
from bel4ed.hydro.transport.mt3d import transport
from bel4ed.processing.target_handle import travelling_particles


def forward_modelling(folder=None):
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
    bel4ed.utils.dirmaker(results_dir)

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
                bel4ed.utils.keep_essential(results_dir)
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
