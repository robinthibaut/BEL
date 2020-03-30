# Import
import os
from os import listdir
from os.path import join as jp, isfile

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats
from sklearn.neighbors import KernelDensity

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

s = 0
true0 = np.load(t_files[s])
forecast0 = np.load(f_files[s])
shape = np.array(true0).shape
# %% extract 0 contours

c0s = [plt.contour(mp.x, mp.y, f, [0]) for f in forecast0]
v = np.array([c0.allsegs[0][0] for c0 in c0s])
# ps = [c0.collections[0].get_paths()[0] for c0 in c0s]
# v = [p.vertices for p in ps]
plt.close()

x = np.hstack([vi[:, 0] for vi in v])
y = np.hstack([vi[:, 1] for vi in v])

xykde = np.vstack([x, y]).T

# %% Kernel density

nn = 2
plt.plot(v[nn][:, 0], v[nn][:, 1], 'o-')
plt.show()

# Sklearn

xmin = 0
xmax = 1500
ymin = 0
ymax = 1000
xgrid = np.arange(xmin, xmax, 5)
ygrid = np.arange(ymin, ymax, 5)
X, Y = np.meshgrid(xgrid, ygrid)
xy = np.vstack([X.ravel(), Y.ravel()]).T

x0, y0, radius = 1000, 500, 200
r = np.sqrt((xy[:, 0] - x0)**2 + (xy[:, 1] - y0)**2)
inside = r < radius
xyu = xy[inside]


kde = KernelDensity(kernel='gaussian',
                    bandwidth=1.5).fit(xykde)
score = np.exp(kde.score_samples(xyu))
z = np.full(inside.shape, 0, dtype=float)
z[inside] = score

z = np.flipud(z.reshape(shape))
plt.imshow(z,
           extent=(xmin, xmax, ymin, ymax),
           cmap='coolwarm')
plt.contour(mp.x, mp.y, true0, [0], colors='red')
wells_xy = np.load(jp(cwd, 'grid', 'iw.npy'), allow_pickle=True)[:, :2]
plt.plot(wells_xy[:, 0], wells_xy[:, 1], 'ko', alpha=1)
plt.xlim(750, 1200)
plt.ylim(300, 700)
plt.show()

levels = np.linspace(-500, z.max(), 25)
plt.contourf(X, Y, z, cmap='coolwarm')
plt.show()


# Seaborn

wells_xy = np.load(jp(cwd, 'grid', 'iw.npy'), allow_pickle=True)[:, :2]
sns.kdeplot(x, y, cmap="coolwarm", shade=True, shade_lowest=False, cbar=True)
plt.contour(mp.x, mp.y, true0, [0], colors='red')
plt.plot(wells_xy[:, 0], wells_xy[:, 1], 'co', alpha=1)
plt.xlim(750, 1200)
plt.ylim(300, 700)
plt.savefig(jp(bel_dir, 'comp.pdf'))
plt.show()

# Scipy
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


