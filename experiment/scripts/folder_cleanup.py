#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os

import experiment._utils as fops
import experiment.utils
from experiment.core import Setup


def cleanup():
    res_tree = Setup.Directories.hydro_res_dir
    # r=root, d=directories, f = files
    for r, d, f in os.walk(res_tree, topdown=False):
        if r != res_tree:
            experiment.utils.keep_essential(r)
            experiment.utils.remove_bad_bkt(r)
            experiment.utils.remove_incomplete(r)
    print('Folders cleaned up')


def filter_file(crit):
    res_tree = Setup.Directories.hydro_res_dir
    for r, d, f in os.walk(res_tree, topdown=False):
        if r != res_tree:
            experiment.utils.remove_bad_bkt(r)
            experiment.utils.remove_incomplete(r, crit=crit)
    print(f'Folders filtered based on {crit}')


def spare_me():
    res_tree = Setup.Directories.hydro_res_dir
    for r, d, f in os.walk(res_tree, topdown=False):
        if r != res_tree:
            experiment.utils.keep_essential(r)
            experiment.utils.remove_bad_bkt(r)


if __name__ == '__main__':
    cleanup()
    filter_file('pz.npy')
    # spare_me()
