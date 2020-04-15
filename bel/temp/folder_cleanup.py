import os
from os.path import join as jp

from bel.toolbox.file_ops import FileOps


def cleanup():
    fo = FileOps()
    cwd = os.getcwd()
    res_tree = jp('..', 'hydro', 'results')
    # r=root, d=directories, f = files


if __name__ == '__main__':
    cleanup()