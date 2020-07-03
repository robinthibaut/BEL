#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import shutil
from collections import deque
from os.path import join as jp

import flopy
import numpy as np

from experiment.base.inventory import MySetup


def datread(file=None, start=0, end=None):
    # end must be set to None and NOT -1
    """Reads space separated dat file"""
    with open(file, 'r') as fr:
        lines = np.copy(fr.readlines())[start:end]
        try:
            op = np.array([list(map(float, line.split())) for line in lines])
        except ValueError:
            op = [line.split() for line in lines]
    return op


def folder_reset(folder):
    """Deletes files out of folder"""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def dirmaker(dird):
    """
    Given a folder path, check if it exists, and if not, creates it
    :param dird: path
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


def load_flow_model(nam_file, exe_name='', model_ws=''):
    flow_loader = flopy.modflow.mf.Modflow.load

    return flow_loader(f=nam_file, exe_name=exe_name, model_ws=model_ws)


def load_transport_model(nam_file, modflowmodel, exe_name='', model_ws='', ftl_file='mt3d_link.ftl',
                         version='mt3d-usgs'):
    transport_loader = flopy.mt3d.Mt3dms.load
    transport_reloaded = transport_loader(f=nam_file, version=version, modflowmodel=modflowmodel,
                                          exe_name=exe_name, model_ws=model_ws)
    transport_reloaded.ftlfilename = ftl_file

    return transport_reloaded


def remove_sd(res_tree):
    """

    :param res_tree: Path directing to the folder containing the directories of results
    :return:
    """
    for r, d, f in os.walk(res_tree, topdown=False):
        # Adds the data files to the lists, which will be loaded later
        if 'sd.npy' in f:
            os.remove(jp(r, 'sd.npy'))


def remove_incomplete(res_tree):
    """

    :param res_tree: Path directing to the folder containing the directories of results
    :return:
    """
    for r, d, f in os.walk(res_tree, topdown=False):
        # Adds the data files to the lists, which will be loaded later
        if 'bkt.npy' not in f or 'hk.npy' not in f or 'pz.npy' not in f:
            shutil.rmtree(r)


def keep_essential(res_dir):
    """
    Deletes everything in a simulation folder except specific files.
    :param res_dir: Path to the folder containing results
    :return:
    """
    for the_file in os.listdir(res_dir):
        if not the_file.endswith('.npy') and not the_file.endswith('.py') and not the_file.endswith('.xy'):
            file_path = os.path.join(res_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)


def remove_bad(res_tree):
    """

    :param res_tree: Path directing to the folder containing the directories of results
    :return:
    """
    for r, d, f in os.walk(res_tree, topdown=False):
        # Adds the data files to the lists, which will be loaded later
        if 'mt3d_link.ftl' in f:
            if r != res_tree:  # Make sure to not delete the main results directory !
                print('removed 1 folder')
                shutil.rmtree(r)


def remove_bkt(res_dir):
    """
    Loads all breakthrough curves from the results and delete folder in case
    the max computed concentration is > 1.
    :param res_dir:
    :return:
    """
    bkt_files = []  # Breakthrough curves files
    # r=root, d=directories, f = files
    roots = []
    for r, d, f in os.walk(res_dir, topdown=False):
        # Adds the data files to the lists, which will be loaded later
        if 'bkt.npy' in f:
            bkt_files.append(jp(r, 'bkt.npy'))
            roots.append(r)
    tpt = list(map(np.load, bkt_files))
    rm = []  # Will contain indices to remove
    for i in range(len(tpt)):
        for j in range(len(tpt[i])):
            if max(tpt[i][j][:, 1]) > 1:  # Check results files whose max computed head is > 1 and removes them
                rm.append(i)
                break
    for index in sorted(rm, reverse=True):
        shutil.rmtree(roots[index])


def load_res(res_dir=None, roots=None, test_roots=None, d=False, h=False):
    """
    Loads results from main results folder.
    :param test_roots: Specified roots for testing
    :param roots: Specified roots for training
    :param res_dir: main directory containing results sub-directories
    :return: tp, sd, roots
    """

    # If no res_dir specified, then uses default
    if res_dir is None:
        res_dir = MySetup.Directories.hydro_res_dir

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
            roots = [roots]

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
