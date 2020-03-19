# Import
import os
from os import listdir
from os.path import join as jp, isfile

import numpy as np
import matplotlib.pyplot as plt

# from diavatly import blocks_from_rc, model_map
from MyToolbox import Plot, MeshOps

plt.style.use('dark_background')
mp = Plot()
mo = MeshOps()

# Directories
cwd = os.getcwd()
wdir = jp(cwd, 'grid')
res_dir = jp(cwd, 'temp', 'forecasts')

f_files = [jp(res_dir, f) for f in listdir(res_dir) if isfile(jp(res_dir, f)) and 'forecasts' in f]
t_files = [jp(res_dir, f) for f in listdir(res_dir) if isfile(jp(res_dir, f)) and 'true' in f]

s = 1
true0 = np.load(t_files[s])
forecast0 = np.load(f_files[s])[:500]

super_mean = np.mean(forecast0, axis=0)
diff_stacked = np.subtract(true0, super_mean)
diff_stacked = super_mean
ll = np.expand_dims(diff_stacked, axis=0)

mp.whp(np.expand_dims(true0, axis=0),
       colors='black',
       lw=1,
       bkg_field_array=ll[0],
       vmin=-25,
       vmax=25,
       show=True)


