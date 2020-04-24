#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import bel.toolbox.file_ops as fops


def cleanup():
    # cwd = os.getcwd()
    res_tree = jp('..', 'hydro', 'results')
    # r=root, d=directories, f = files
    fops.remove_incomplete(res_tree)
    for r, _, _ in os.walk(res_tree, topdown=False):
        if r != res_tree:
            fops.keep_essential(r)
            fops.remove_bkt(r)


if __name__ == '__main__':
    cleanup()
