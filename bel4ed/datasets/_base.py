#  Copyright (c) 2021. Robin Thibaut, Ghent University
import os
import shutil
from os.path import dirname, join
from typing import List

import flopy
import numpy as np
import pandas as pd
from loguru import logger

from bel4ed.config import Setup
from bel4ed.utils import data_read

__all__ = [
    "load_dataset",
    "load_flow_model",
    "load_transport_model",
    "remove_sd",
    "remove_incomplete",
    "keep_essential",
    "remove_bad_bkt",
    "data_loader",
    "i_am_root",
    "cleanup",
    "filter_file",
    "spare_me",
]


def load_dataset():
    module_path = os.path.dirname(__file__)
    predictor = pd.read_pickle(join(module_path, "data", "predictor.pkl"))
    target = pd.read_pickle(join(module_path, "data", "target.pkl"))

    return predictor, target


def load_flow_model(nam_file: str, exe_name: str = "", model_ws: str = ""):
    """
    Loads a modflow model.
    :param nam_file: str: Path to the 'nam' file.
    :param exe_name: str: Path to the modflow exe file.
    :param model_ws: str: Working directory.
    :return:
    """
    flow_loader = flopy.modflow.mf.Modflow.load

    return flow_loader(f=nam_file, exe_name=exe_name, model_ws=model_ws)


def load_transport_model(
        nam_file: str,
        modflowmodel,
        exe_name: str = "",
        model_ws: str = "",
        ftl_file: str = "mt3d_link.ftl",
        version: str = "mt3d-usgs",
):
    """
    Loads a transport model.

    :param nam_file: str: Path to the 'nam' file.
    :param modflowmodel: Modflow model object.
    :param exe_name: str: Path to the mt3d exe file.
    :param model_ws: str: Working directory.
    :param ftl_file: str: Path to the 'ftl' file.
    :param version: str: Mt3dms version.
    :return:
    """
    transport_loader = flopy.mt3d.Mt3dms.load
    transport_reloaded = transport_loader(
        f=nam_file,
        version=version,
        modflowmodel=modflowmodel,
        exe_name=exe_name,
        model_ws=model_ws,
    )
    transport_reloaded.ftlfilename = ftl_file

    return transport_reloaded


def remove_sd(res_tree: str):
    """
    Deletes signed distance file out of sub-folders of folder res_tree.
    :param res_tree: str: Path directing to the folder containing the directories of results
    :return:
    """
    for r, d, f in os.walk(res_tree, topdown=False):
        # Adds the data files to the lists, which will be loaded later
        if "sd.npy" in f:
            os.remove(join(r, "sd.npy"))


def remove_incomplete(res_tree: str, crit: str = None):
    """

    :param res_tree: str: Path directing to the folder containing the directories of results.
    :param crit: str: Name of a file according to which delete folder if not present in said folder.
    :return:
    """

    if crit is None:
        ck = np.array(
            [os.path.isfile(join(res_tree, d)) for d in Setup.Files.output_files]
        )
    else:
        ck = np.array([os.path.isfile(join(res_tree, crit))])

    opt = ck.all()

    if not opt:
        try:
            shutil.rmtree(res_tree)
        except FileNotFoundError:
            pass


def keep_essential(res_dir: str):
    """
    Deletes everything in a simulation folder except specific files.
    :param res_dir: Path to the folder containing results.
    """
    for the_file in os.listdir(res_dir):
        if (
                not the_file.endswith(".npy")
                and not the_file.endswith(".py")
                and not the_file.endswith(".xy")
                and not the_file.endswith(".sgems")
        ):

            file_path = os.path.join(res_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(e)


def remove_bad_bkt(res_dir: str):
    """
    Loads all breakthrough curves from the results and delete folder in case
    the max computed concentration is > 1.
    :param res_dir: str: Path to result directory.
    """
    bkt_files = []  # Breakthrough curves files
    # r=root, d=directories, f = files
    roots = []
    for r, d, f in os.walk(res_dir, topdown=False):
        # Adds the data files to the lists, which will be loaded later
        if "bkt.npy" in f:
            bkt_files.append(join(r, "bkt.npy"))
            roots.append(r)
    if bkt_files:
        tpt = list(map(np.load, bkt_files))
        rm = []  # Will contain indices to remove
        for i in range(len(tpt)):
            for j in range(len(tpt[i])):
                # Check results files whose max computed head is > 1 and removes them
                if max(tpt[i][j][:, 1]) > 1:
                    rm.append(i)
                    break
        for index in sorted(rm, reverse=True):
            shutil.rmtree(roots[index])


def data_loader(
        res_dir: str = None,
        roots: List[str] = None,
        test_roots: List[str] = None,
        d: bool = False,
        h: bool = False,
):
    """
    Loads results from main results folder.

    :param roots: Specified roots for training.
    :param test_roots: Specified roots for testing.
    :param res_dir: main directory containing results sub-directories.
    :param d: bool: Flag to load predictor.
    :param h: bool: Flag to load target.
    :return: tp, sd, roots
    """

    # If no res_dir specified, then uses default
    if res_dir is None:
        res_dir = Setup.Directories.hydro_res_dir

    bkt_files = []  # Breakthrough curves files
    sd_files = []  # Signed-distance files
    hk_files = []  # Hydraulic conductivity files
    # r=root, d=directories, f = files

    if roots is None and test_roots is not None:
        if not isinstance(test_roots, (list, tuple)):
            roots = [test_roots]
        else:
            roots = test_roots
    else:
        if not isinstance(roots, (list, tuple)):
            roots: list = [roots]

    [bkt_files.append(join(res_dir, r, "bkt.npy")) for r in roots]
    [sd_files.append(join(res_dir, r, "pz.npy")) for r in roots]
    [hk_files.append(join(res_dir, r, "hk0.npy")) for r in roots]

    if d:
        tpt = list(map(np.load, bkt_files))  # Re-load transport curves
    else:
        tpt = None
    if h:
        sd = np.array(list(map(np.load, sd_files)))  # Load signed distance
    else:
        sd = None

    return tpt, sd, roots


def i_am_root(
        training_file: str = None,
        test_file: str = None,
):
    if training_file is None:
        training_file = os.path.join(Setup.Directories.storage_dir, "roots.dat")
    if test_file is None:
        test_file = os.path.join(Setup.Directories.storage_dir, "test_roots.dat")

    # List directories in forwards folder
    training_roots = data_read(training_file)
    training_roots = [item for sublist in training_roots for item in sublist]

    test_roots = data_read(test_file)
    test_roots = [item for sublist in test_roots for item in sublist]

    return training_roots, test_roots


def cleanup():
    res_tree = Setup.Directories.hydro_res_dir
    # r=root, d=directories, f = files
    for r, d, f in os.walk(res_tree, topdown=False):
        if r != res_tree:
            keep_essential(r)
            remove_bad_bkt(r)
            remove_incomplete(r)
    logger.info("Folders cleaned up")


def filter_file(crit: str):
    res_tree = Setup.Directories.hydro_res_dir
    for r, d, f in os.walk(res_tree, topdown=False):
        if r != res_tree:
            remove_bad_bkt(r)
            remove_incomplete(r, crit=crit)
    logger.info(f"Folders filtered based on {crit}")


def spare_me():
    res_tree = Setup.Directories.hydro_res_dir
    for r, d, f in os.walk(res_tree, topdown=False):
        if r != res_tree:
            keep_essential(r)
            remove_bad_bkt(r)
