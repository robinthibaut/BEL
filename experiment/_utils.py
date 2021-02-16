#  Copyright (c) 2021. Robin Thibaut, Ghent University
import itertools
import os
import shutil
from typing import List
from os.path import join as jp

import flopy
import joblib
import numpy as np

from experiment._core import setup


def data_read(file: str = None,
              start: int = 0,
              end: int = None):
    # end must be set to None and NOT -1
    """Reads space-separated dat file"""
    with open(file, 'r') as fr:
        lines = np.copy(fr.readlines())[start:end]
        try:
            op = np.array([list(map(float, line.split())) for line in lines], dtype=object)
        except ValueError:
            op = [line.split() for line in lines]
    return op


def folder_reset(folder: str,
                 exceptions: list = None):
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
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
    except FileNotFoundError:
        pass


def empty_figs(root: str):
    """Empties figure folders"""

    if isinstance(root, (list, tuple)):
        if len(root) > 1:
            print('Input error')
            return
        else:
            root = root[0]

    subdir = os.path.join(setup.directories.forecasts_dir, root)
    listme = os.listdir(subdir)
    folders = list(filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))

    for f in folders:
        # pca
        folder_reset(os.path.join(subdir, f, 'pca'))
        # cca
        folder_reset(os.path.join(subdir, f, 'cca'))
        # uq
        folder_reset(os.path.join(subdir, f, 'uq'))
        # data
        folder_reset(os.path.join(subdir, f, 'cca'))


def dirmaker(dird: str):
    """
    Given a folder path, check if it exists, and if not, creates it.
    :param dird: str: Directory path.
    :return:
    """
    try:
        if not os.path.exists(dird):
            os.makedirs(dird)
            return 0
        else:
            return 1
    except Exception as e:
        print(e)
        return 0


def load_flow_model(nam_file: str,
                    exe_name: str = '',
                    model_ws: str = ''):
    """
    Loads a modflow model.
    :param nam_file: str: Path to the 'nam' file.
    :param exe_name: str: Path to the modflow exe file.
    :param model_ws: str: Working directory.
    :return:
    """
    flow_loader = flopy.modflow.mf.Modflow.load

    return flow_loader(f=nam_file, exe_name=exe_name, model_ws=model_ws)


def load_transport_model(nam_file: str, modflowmodel,
                         exe_name: str = '',
                         model_ws: str = '',
                         ftl_file: str = 'mt3d_link.ftl',
                         version: str = 'mt3d-usgs'):
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
    transport_reloaded = transport_loader(f=nam_file, version=version, modflowmodel=modflowmodel,
                                          exe_name=exe_name, model_ws=model_ws)
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
        if 'sd.npy' in f:
            os.remove(jp(r, 'sd.npy'))


def remove_incomplete(res_tree: str, crit: str = None):
    """

    :param res_tree: str: Path directing to the folder containing the directories of results.
    :param crit: str: Name of a file according to which delete folder if not present in said folder.
    :return:
    """

    if crit is None:
        ck = np.array([os.path.isfile(jp(res_tree, d)) for d in setup.files.output_files])
    else:
        ck = np.array([os.path.isfile(jp(res_tree, crit))])

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
        if not the_file.endswith('.npy') \
                and not the_file.endswith('.py') \
                and not the_file.endswith('.xy') \
                and not the_file.endswith('.sgems'):

            file_path = os.path.join(res_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)


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
        if 'bkt.npy' in f:
            bkt_files.append(jp(r, 'bkt.npy'))
            roots.append(r)
    if bkt_files:
        tpt = list(map(np.load, bkt_files))
        rm = []  # Will contain indices to remove
        for i in range(len(tpt)):
            for j in range(len(tpt[i])):
                if max(tpt[i][j][:, 1]) > 1:  # Check results files whose max computed head is > 1 and removes them
                    rm.append(i)
                    break
        for index in sorted(rm, reverse=True):
            shutil.rmtree(roots[index])


def data_loader(res_dir: str = None,
                roots: List[str] = None,
                test_roots: List[str] = None,
                d: bool = False,
                h: bool = False):
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
        res_dir = setup.directories.hydro_res_dir

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

    [bkt_files.append(jp(res_dir, r, 'bkt.npy')) for r in roots]
    [sd_files.append(jp(res_dir, r, 'pz.npy')) for r in roots]
    [hk_files.append(jp(res_dir, r, 'hk0.npy')) for r in roots]

    if d:
        tpt = list(map(np.load, bkt_files))  # Re-load transport curves
    else:
        tpt = None
    if h:
        sd = np.array(list(map(np.load, sd_files)))  # Load signed distance
    else:
        sd = None

    return tpt, sd, roots


def combinator(combi):
    """Given a n-sized 1D array, generates all possible configurations, from size 1 to n-1.
    'None' will indicate to use the original combination.
    """
    cb = [list(itertools.combinations(combi, i)) for i in range(1, combi[-1] + 1)]  # Get all possible wel combinations
    cb = [item for sublist in cb for item in sublist][::-1]  # Flatten and reverse to get all combination at index 0.
    return cb


def reload_trained_model(root: str,
                         well: str,
                         sample_n: int = 0):
    base_dir = jp(setup.directories.forecasts_dir, 'base')
    res_dir = jp(setup.directories.forecasts_dir, root, well, 'obj')

    f_names = list(map(lambda fn: jp(res_dir, f'{fn}.pkl'), ['cca', 'd_pca', 'post']))
    cca_operator, d_pco, post = list(map(joblib.load, f_names))

    h_pco = joblib.load(jp(base_dir, 'h_pca.pkl'))
    h_pred = np.load(jp(base_dir, 'roots_whpa', f'{root}.npy'))

    # Inspect transformation between physical and PC space
    dnc0 = d_pco.n_pc_cut
    hnc0 = h_pco.n_pc_cut

    # Cut desired number of PC components
    d_pc_training, d_pc_prediction = d_pco.comp_refresh(dnc0)
    h_pco.test_transform(h_pred, test_root=[root])
    h_pc_training, h_pc_prediction = h_pco.comp_refresh(hnc0)

    # CCA scores
    d_cca_training, h_cca_training = cca_operator.x_scores_, cca_operator.y_scores_

    d, h = d_cca_training.T, h_cca_training.T

    d_obs = d_pc_prediction[sample_n]
    h_obs = h_pc_prediction[sample_n]

    # # Transform to CCA space and transpose
    d_cca_prediction, h_cca_prediction = cca_operator.transform(d_obs.reshape(1, -1),
                                                                h_obs.reshape(1, -1))

    #  Watch out for the transpose operator.
    h2 = h.copy()
    d2 = d.copy()
    tfm1 = post.normalize_h
    h = tfm1.transform(h2.T)
    h = h.T
    h_cca_prediction = tfm1.transform(h_cca_prediction)
    h_cca_prediction = h_cca_prediction.T

    tfm2 = post.normalize_d
    d = tfm2.transform(d2.T)
    d = d.T
    d_cca_prediction = tfm2.transform(d_cca_prediction)
    d_cca_prediction = d_cca_prediction.T

    return d, h, d_cca_prediction, h_cca_prediction, post, cca_operator
