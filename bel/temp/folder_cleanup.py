import os
from os.path import join as jp
import shutil

cwd = os.getcwd()
res_dir = jp('..', 'hydro', 'results')
# r=root, d=directories, f = files
for r, d, f in os.walk(res_dir, topdown=False):
    # Adds the data files to the lists, which will be loaded later
    if 'sd.npy' in f:
        os.remove(jp(r, 'sd.npy'))

for r, d, f in os.walk(res_dir, topdown=False):
    # Adds the data files to the lists, which will be loaded later
    if 'bkt.npy' in f and 'pz.npy' in f and 'hk0.npy' in f:
        pass
    else:  # If one of the files is missing, deletes the sub-folder
        try:
            if r != res_dir:  # Make sure to not delete the main results directory !
                shutil.rmtree(r)
        except TypeError:
            pass
