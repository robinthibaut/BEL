import os
from os.path import join as jp
import shutil

# cwd = os.getcwd()
# res_tree = jp('..', 'hydro', 'results')
# r=root, d=directories, f = files


def remove_sd(res_tree):
    """

    :param res_tree: Path directing to the folder containing the directories of results
    :return:
    """
    for r, d, f in os.walk(res_tree, topdown=False):
        # Adds the data files to the lists, which will be loaded later
        if 'sd.npy' in f:
            os.remove(jp(r, 'sd.npy'))


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
