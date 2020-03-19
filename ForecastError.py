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


lol = np.sum(array500, axis=0)
lol = lol - lol.min()
lol = lol/lol.max() - .5
lol = lol.reshape(1, 200, 300)

mp.whp(lol, bkg_field_array=lol[0], vmin=0.3, vmax=.45, show=True)

plt.imshow(array500[0], cmap='coolwarm')
plt.colorbar()
plt.show()

plt.imshow(lol[0], cmap='coolwarm')

plt.colorbar()
plt.show()

rows = np.ones(200)*5
cols = np.ones(300)*5

blocks = blocks_from_rc(rows, cols)
model_map(blocks, lol.flatten(), log=0)
plt.show()