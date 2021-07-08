#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
import shutil
import time
import uuid
from os.path import join as jp, isfile

import numpy as np
from loguru import logger
from pysgems.algo.sgalgo import XML
from pysgems.dis.sgdis import Discretize
from pysgems.io.sgio import PointSet
from pysgems.sgems import sg

import bel4ed.utils as utils
from bel4ed.config import Setup, Machine
from bel4ed.datasets import keep_essential
from bel4ed.hydro.backtracking.modpath import backtrack
from bel4ed.hydro.flow.modflow import flow
from bel4ed.hydro.transport.mt3d import transport
from bel4ed.preprocessing import travelling_particles
from bel4ed.utils import dirmaker

__all__ = ["forward_modelling"]


def _log_transform(f, k_mean: float, k_std: float):
    """
    Transforms the values of the statistical_simulation simulations into meaningful data.
    :param: f: np.array: Simulation output = Hk field
    :param: k_mean: float: Mean of the Hk field
    :param: k_std: float: Standard deviation of the Hk field
    """
    # TODO: Move this to pysgems package

    ff = f * k_std + k_mean

    return 10 ** ff


def sgsim(model_ws: str, grid_dir: str, wells_hk: list = None, save: bool = True):
    # TODO: Move this to pysgems package
    """
    Perform sequential gaussian simulation to generate K fields.
    :param model_ws: str: Working directory
    :param grid_dir: str: Grid directory
    :param wells_hk: List[float]: K values at wells
    :return:
    """
    # Initiate sgems pjt
    pjt = sg.Sgems(
        project_name="sgsim", project_wd=grid_dir, res_dir=model_ws, verbose=False
    )

    # Load hard data point set

    data_dir = grid_dir
    dataset = "wells.eas"
    file_path = jp(data_dir, dataset)

    hd = PointSet(project=pjt, pointset_path=file_path)

    k_params = Setup.ModelParameters
    k_min, k_max, k_std = (
        k_params.k_min,
        k_params.k_max,
        k_params.k_std,
    )

    # Hydraulic conductivity exponent mean between x and y.
    k_mean = np.random.uniform(k_min, k_max)

    if wells_hk is None:
        # Fix hard data values at wells location
        hku = k_max + np.random.rand(len(hd.dataframe))
    else:
        hku = wells_hk

    if not os.path.exists(jp(model_ws, Setup.Files.sgems_file)):
        hd.dataframe["hd"] = hku
        hd.export_01(["hd"])  # Exports modified dataset in binary

    # Generate grid. Grid dimensions can automatically be generated based on the data points
    # unless specified otherwise, but cell dimensions dx, dy, (dz) must be specified
    gd = Setup.GridDimensions()
    Discretize(
        project=pjt,
        dx=gd.dx,
        dy=gd.dy,
        xo=gd.xo,
        yo=gd.yo,
        x_lim=gd.x_lim,
        y_lim=gd.y_lim,
    )

    # Get sgems grid centers coordinates:
    x = np.cumsum(pjt.dis.along_r) - pjt.dis.dx / 2
    y = np.cumsum(pjt.dis.along_c) - pjt.dis.dy / 2
    xv, yv = np.meshgrid(x, y, sparse=False, indexing="xy")
    centers = np.stack((xv, yv), axis=2).reshape((-1, 2))

    if os.path.exists(jp(model_ws, "hk0.npy")):
        hk0 = np.load(jp(model_ws, "hk0.npy"))
        return hk0, centers

    # Load your algorithm xml file in the 'algorithms' folder.
    # dir_path = os.path.abspath(__file__ + "/..")
    # algo_dir = jp(dir_path, "")
    al = XML(project=pjt, algo_dir=Setup.Directories.algo_dir)
    al.xml_reader("bel_sgsim")

    # Modify xml below:
    al.xml_update("Seed", "value", str(np.random.randint(1e9)), show=False)
    # Structural uncertainty
    r_max = round(np.random.uniform(2, 4) * 100)
    al.xml_update("Variogram//structure_1//ranges", "max", str(r_max), show=False)
    y_tilt = round(np.random.uniform(60, 120), 1)
    al.xml_update("Variogram//structure_1//angles", "x", str(y_tilt), show=False)

    # Write python script
    pjt.write_command()

    # Run sgems
    pjt.run()

    opl = jp(model_ws, "results.grid")  # Output file location.

    # Grid information directly derived from the output file.
    matrix = utils.data_read(opl, start=3)

    if matrix.size == 0:
        logger.warning("Empty matrix")
        return 0, 0

    matrix = np.where(matrix == -9966699, np.nan, matrix)

    tf = np.vectorize(_log_transform)

    k_std = np.random.uniform(np.sqrt(0.1), np.sqrt(0.3))
    # Transform values from log10
    matrix = tf(matrix, k_mean, k_std)  # Apply function to results

    matrix = matrix.reshape((pjt.dis.nrow, pjt.dis.ncol))  # reshape - assumes 2D !
    matrix = np.flipud(matrix)  # Flip to correspond to sgems

    if save:
        np.save(jp(model_ws, "hk0"), matrix)  # Save the un-discretized hk grid

    return matrix, centers


def forward_modelling(args, **kwargs):
    """Data collection"""

    if args:
        kwargs = args

    folder = kwargs["folder"]

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
    dirmaker(results_dir)

    logger.info(f"fwd {res_dir}")
    # Check if forwards have already been computed
    of = Setup.Files.output_files
    opt = np.array(
        [os.path.isfile(jp(results_dir, d)) for d in of]
    )
    override = kwargs["override"]

    if not opt.all() or override:
        # Resets folder
        # fops.folder_reset(results_dir, exceptions=MySetup.Files.sgems_family)

        start_fwd = time.time()

        # Statistical simulation
        hk_array, xy_dummy = sgsim(model_ws=results_dir, grid_dir=grid_dir)

        # Run Flow
        flow_model = None
        if kwargs["flow"] or not isfile(jp(results_dir, of.heads_file)):
            flow_model = flow(
                exe_name=exe_name_mf,
                model_ws=results_dir,
                grid_dir=grid_dir,
                hk_array=hk_array,
                xy_dummy=xy_dummy,
            )
        # Run Transport and Backtracking
        if flow_model and (kwargs["transport"] or not isfile(jp(results_dir, of.predictor_file))):
            transport(
                modflowmodel=flow_model,
                exe_name=exe_name_mt,
                grid_dir=grid_dir,
                save_ucn=kwargs["ucn"],
            )
            # Run Modpath
        if flow_model and (kwargs["backtrack"] or not isfile(jp(results_dir, of.target_file))):
            end_points = backtrack(flow_model, exe_name_mp)
            # Compute particle delineation to compute signed distance later on
            # indices of the vertices of the final protection zone
            delineation = travelling_particles(end_points)
            # using TSP algorithm
            pzs = end_points[delineation]  # x-y coordinates protection zone
            np.save(jp(results_dir, "pz"), pzs)  # Save those

        # Deletes everything except final results
        hl = (time.time() - start_fwd) // 60
        logger.info(f"done in {hl} min")

        if kwargs["flush"]:
            keep_essential(results_dir)
        # else:
        #     shutil.rmtree(results_dir)
        #     logger.info(f"terminated {res_dir}")
        #     return 0
    else:
        logger.info(f"pass {res_dir}")

    return results_dir
