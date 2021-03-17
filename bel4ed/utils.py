import itertools
import os
import shutil
from os.path import join as jp
from typing import List

import joblib
import types
import numpy as np
import pandas as pd
from loguru import logger

from .config import Setup

FLOAT_DTYPES = (np.float64, np.float32, np.float16)
Root = List[str]
Combination = List[List[int]]
Function = types.FunctionType


def flatten_array(arr: np.array) -> np.array:
    arr_flat = np.array([item for sublist in arr for item in sublist])
    return arr_flat.reshape(1, -1)


def i_am_framed(array: np.array, ids: dict = None, flat: bool = True) -> pd.DataFrame:
    """Build a panda's dataframe that contains the flattened samples, their id's and the original shape"""
    # Remember original shape
    physical_shape = array.shape

    # Flatten for dimension reduction
    if flat:
        array = flatten_array(array)

    # Save DataFrames
    # Initiate array
    training_df_target = pd.DataFrame(data=array)
    # Set id's
    if ids is not None:
        for key in ids:
            training_df_target[key] = ids[key]
        training_df_target.set_index([k for k in ids], inplace=True)
    # Save original shape as a dataframe attribute
    training_df_target.attrs["physical_shape"] = physical_shape

    return training_df_target


def data_read(file: str = None, start: int = 0, end: int = None):
    # end must be set to None and NOT -1
    """Reads space-separated dat file"""
    with open(file, "r") as fr:
        lines = np.copy(fr.readlines())[start:end]
        try:
            op = np.array(
                [list(map(float, line.split())) for line in lines], dtype=object
            )
        except ValueError:
            op = [line.split() for line in lines]
    return op


def folder_reset(folder: str, exceptions: list = None):
    """Deletes files in folder"""
    if not isinstance(exceptions, (list, tuple)):
        exceptions = [exceptions]
    try:
        for filename in os.listdir(folder):
            if filename not in exceptions:
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.warning("Failed to delete %s. Reason: %s" % (file_path, e))
    except FileNotFoundError:
        pass


def empty_figs(root: str):
    """Empties figure folders"""

    if isinstance(root, (list, tuple)):
        if len(root) > 1:
            logger.error("Input error")
            return
        else:
            root = root[0]

    subdir = os.path.join(Setup.Directories.forecasts_dir, root)
    listme = os.listdir(subdir)
    folders = list(filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))

    for f in folders:
        # pca
        folder_reset(os.path.join(subdir, f, "pca"))
        # cca
        folder_reset(os.path.join(subdir, f, "cca"))
        # uq
        folder_reset(os.path.join(subdir, f, "uq"))
        # data
        folder_reset(os.path.join(subdir, f, "cca"))


def dirmaker(dird: str, erase: bool = False):
    """
    Given a folder path, check if it exists, and if not, creates it.
    :param dird: str: Directory path.
    :param erase: bool: Whether to delete existing folder or not.
    :return:
    """
    try:
        if not os.path.exists(dird):
            os.makedirs(dird)
            return 0
        else:
            if erase:
                shutil.rmtree(dird)
                os.makedirs(dird)
            return 1
    except Exception as e:
        logger.warning(e)
        return 0


def combinator(combi):
    """Given a n-sized 1D array, generates all possible configurations, from size 1 to n-1.
    'None' will indicate to use the original combination.
    """
    cb = [
        list(itertools.combinations(combi, i)) for i in range(1, combi[-1] + 1)
    ]  # Get all possible wel combinations
    # Flatten and reverse to get all combination at index 0.
    cb = [item for sublist in cb for item in sublist][::-1]
    return cb


def reload_trained_model(root: str, well: str, sample_n: int = 0):
    base_dir = jp(Setup.Directories.forecasts_dir, "base")
    res_dir = jp(Setup.Directories.forecasts_dir, root, well, "obj")

    f_names = list(map(lambda fn: jp(res_dir, f"{fn}.pkl"), ["cca", "d_pca", "post"]))
    cca_operator, d_pco, post = list(map(joblib.load, f_names))

    h_pco = joblib.load(jp(base_dir, "h_pca.pkl"))
    h_pred = np.load(jp(base_dir, "roots_whpa", f"{root}.npy"))

    # Inspect transformation between physical and PC space
    dnc0 = d_pco.n_pc_cut
    hnc0 = h_pco.n_pc_cut

    # Cut desired number of PC components
    d_pc_training, d_pc_prediction = d_pco.comp_refresh(dnc0)
    h_pco.test_transform(h_pred, test_roots=[root])
    h_pc_training, h_pc_prediction = h_pco.comp_refresh(hnc0)

    # CCA scores
    d_cca_training, h_cca_training = cca_operator.x_scores_, cca_operator.y_scores_

    d, h = d_cca_training.T, h_cca_training.T

    d_obs = d_pc_prediction[sample_n]
    h_obs = h_pc_prediction[sample_n]

    # # Transform to CCA space and transpose
    d_cca_prediction, h_cca_prediction = cca_operator.fit_transform(
        d_obs.reshape(1, -1), h_obs.reshape(1, -1)
    )

    #  Watch out for the transpose operator.
    h2 = h.copy()
    d2 = d.copy()
    tfm1 = post.X_pipeline
    h = tfm1.fit_transform(h2.T)
    h = h.T
    h_cca_prediction = tfm1.fit_transform(h_cca_prediction)
    h_cca_prediction = h_cca_prediction.T

    tfm2 = post.Y_pipeline
    d = tfm2.fit_transform(d2.T)
    d = d.T
    d_cca_prediction = tfm2.fit_transform(d_cca_prediction)
    d_cca_prediction = d_cca_prediction.T

    return d, h, d_cca_prediction, h_cca_prediction, post, cca_operator
