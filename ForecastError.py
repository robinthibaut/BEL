# Import
import os
from os import listdir
from os.path import join as jp, isfile

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats

from diavatly import blocks_from_rc, model_map
from MyToolbox import Plot, MeshOps

plt.style.use('dark_background')
mp = Plot()
mo = MeshOps()

# Directories
cwd = os.getcwd()
wdir = jp(cwd, 'grid')
bel_dir = jp(cwd, 'bel', '43654364-80b7-4916-82c5-a08b0013631d')
res_dir = jp(bel_dir, 'objects')

f_files = [jp(res_dir, f) for f in listdir(res_dir) if isfile(jp(res_dir, f)) and 'forecasts' in f]
t_files = [jp(res_dir, f) for f in listdir(res_dir) if isfile(jp(res_dir, f)) and 'true' in f]

s = 9
true0 = np.load(t_files[s])
forecast0 = np.load(f_files[s])

# %% extract 0 contours

c0s = [plt.contour(mp.x, mp.y, f, [0]) for f in forecast0]
v = np.array([c0.allsegs[0][0] for c0 in c0s])
# ps = [c0.collections[0].get_paths()[0] for c0 in c0s]
# v = [p.vertices for p in ps]
plt.close()

# %% Kernel density

plt.plot(v[1][:, 0], v[1][:, 1], 'o-')
plt.show()

# Seaborn

wells_xy = np.load(jp(cwd, 'grid', 'iw.npy'), allow_pickle=True)[:, :2]

x = np.hstack([vi[:, 0] for vi in v])
y = np.hstack([vi[:, 1] for vi in v])
sns.kdeplot(x, y, cmap="coolwarm", shade=True, shade_lowest=False)
plt.plot(wells_xy[:, 0], wells_xy[:, 1], 'co', alpha=1)
plt.show()

# Scipy
xykde = np.vstack([x, y]).T
kernel = stats.gaussian_kde(xykde)


# %% mean approach
super_mean = np.mean(forecast0, axis=0)
diff_stacked = np.subtract(true0, super_mean)
diff_stacked = super_mean
ll = np.expand_dims(diff_stacked, axis=0)


mp.whp([true0, super_mean],
       colors='black',
       lw=1,
       vmin=-10,
       vmax=25,
       bkg_field_array=super_mean,
       show=True)


mp.whp(np.expand_dims(true0-super_mean, axis=0),
       colors='black',
       lw=1,
       bkg_field_array=true0-super_mean,
       # vmin=-25,
       # vmax=25,
       show=True)


