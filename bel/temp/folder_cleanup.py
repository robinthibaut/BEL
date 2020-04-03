import os
from os.path import join as jp

cwd = os.getcwd()
res_dir = jp('..', 'hydro', 'results')
# r=root, d=directories, f = files
for r, d, f in os.walk(res_dir, topdown=False):
    # Adds the data files to the lists, which will be loaded later
    if 'sd.npy' in f:
        os.remove(jp(r, 'sd.npy'))
