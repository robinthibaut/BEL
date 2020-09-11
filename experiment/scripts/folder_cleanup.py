#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os

import experiment.toolbox.filesio as fops
from experiment.base.inventory import MySetup


def cleanup():
    # cwd = os.getcwd()
    res_tree = MySetup.Directories.hydro_res_dir
    # r=root, d=directories, f = files
    for r, d, f in os.walk(res_tree, topdown=False):
        if r != res_tree:
            fops.keep_essential(r)
            fops.remove_bkt(r)
            fops.remove_incomplete(r)
    print('Folders cleaned up')


if __name__ == '__main__':
    cleanup()
