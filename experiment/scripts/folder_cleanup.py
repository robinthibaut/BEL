#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os

import experiment.toolbox.filesio as fops
from experiment.base.inventory import MySetup


def cleanup():
    res_tree = MySetup.Directories.hydro_res_dir
    # r=root, d=directories, f = files
    for r, d, f in os.walk(res_tree, topdown=False):
        if r != res_tree:
            fops.keep_essential(r)
            fops.remove_bad_bkt(r)
            fops.remove_incomplete(r)
    print('Folders cleaned up')


def filter_file(crit):
    res_tree = MySetup.Directories.hydro_res_dir
    for r, d, f in os.walk(res_tree, topdown=False):
        if r != res_tree:
            fops.remove_bad_bkt(r)
            fops.remove_incomplete(r, crit=crit)
    print(f'Folders filtered based on {crit}')


def spare_me():
    res_tree = MySetup.Directories.hydro_res_dir
    for r, d, f in os.walk(res_tree, topdown=False):
        if r != res_tree:
            fops.keep_essential(r)


if __name__ == '__main__':
    # cleanup()
    # filter_file('pz.npy')
    spare_me()
