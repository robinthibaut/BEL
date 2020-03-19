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
res_dir = jp(cwd, 'temp', 'forecasts')

f_files = [jp(res_dir, f) for f in listdir(res_dir) if isfile(jp(res_dir, f)) and 'forecasts' in f]
t_files = [jp(res_dir, f) for f in listdir(res_dir) if isfile(jp(res_dir, f)) and 'true' in f]

true0 = np.load(t_files[0])
forecast0 = np.load(t_files[0])


diff_array = np.subtract(true0, forecast0)
