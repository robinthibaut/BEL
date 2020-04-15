import os
from os.path import join as jp

from bel.toolbox.file_ops import FileOps


def cleanup():
    fo = FileOps()
    # cwd = os.getcwd()
    res_tree = jp('..', 'hydro', 'results')
    # r=root, d=directories, f = files
    fo.remove_incomplete(res_tree)
    for r, _, _ in os.walk(res_tree, topdown=False):
        if r != res_tree:
            fo.keep_essential(r)


if __name__ == '__main__':
    cleanup()
