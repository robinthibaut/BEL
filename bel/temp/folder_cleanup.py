import os
from os.path import join as jp
import shutil

cwd = os.getcwd()
res_dir = jp('..', 'hydro', 'results')
# r=root, d=directories, f = files


def remove_sd():
    for r, d, f in os.walk(res_dir, topdown=False):
        # Adds the data files to the lists, which will be loaded later
        if 'sd.npy' in f:
            os.remove(jp(r, 'sd.npy'))


def keep_essential():
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


def remove_bad():
    for r, d, f in os.walk(res_dir, topdown=False):
        # Adds the data files to the lists, which will be loaded later
        if 'mt3d_link.ftl' in f:
            if r != res_dir:  # Make sure to not delete the main results directory !
                print('removed 1 folder')
                shutil.rmtree(r)
