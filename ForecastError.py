# Import
import os
from os.path import join as jp

import numpy as np
import matplotlib.pyplot as plt

import joblib

from sklearn.neighbors import KernelDensity

from MyToolbox import Plot, MeshOps, PosteriorOps

plt.style.use('dark_background')

mp = Plot()
mo = MeshOps()
po = PosteriorOps()

# Directories & files paths
cwd = os.getcwd()
wdir = jp(cwd, 'grid')
bel_dir = jp(cwd, 'bel', 'efe5b47a-a476-402a-909f-32d09a64aa33')
res_dir = jp(bel_dir, 'objects')
fig_dir = jp(bel_dir, 'figures')
fig_pred_dir = jp(fig_dir, 'Predictions')

# Load objects
f_names = list(map(lambda fn: jp(res_dir, fn + '.pkl'), ['cca', 'd_pca', 'h_pca']))
cca, d_pco, h_pco = list(map(joblib.load, f_names))

# Random sample from the posterior
sample_n = 0
forecast_posterior = po.random_sample(sample_n=sample_n,
                                      pca_d=d_pco,
                                      pca_h=h_pco,
                                      cca_obj=cca,
                                      n_posts=3000, add_comp=0)
# Get the true array of the prediction
d_pc_obs = h_pco.dpp[sample_n]
h_true_obs = h_pco.dp[sample_n]
shape = h_pco.raw_data.shape

# Predicting the SD based for a certain number of 'observations'
h_pc_true_pred = cca.predict(d_pc_obs.reshape(1, -1))
# Going back to the original SD dimension and reshape.
h_pred = h_pco.inverse_transform(h_pc_true_pred).reshape(shape[1], shape[2])

# Plot results
ff = jp(fig_pred_dir, '{}_{}.png'.format(sample_n, cca.ncomp))
mp.whp_prediction(forecasts=forecast_posterior,
                  h_true=h_true_obs,
                  h_pred=h_pred,
                  fig_file=ff)
# %% extract 0 contours
c0s = [plt.contour(mp.x, mp.y, f, [0]) for f in forecast_posterior]
plt.close()
v = np.array([c0.allsegs[0][0] for c0 in c0s])
# ps = [c0.collections[0].get_paths()[0] for c0 in c0s]
# v = [p.vertices for p in ps]

x = np.hstack([vi[:, 0] for vi in v])
y = np.hstack([vi[:, 1] for vi in v])

xykde = np.vstack([x, y]).T

# %% Kernel density

# Scatter plot vertices
# nn = sample_n
# plt.plot(v[nn][:, 0], v[nn][:, 1], 'o-')
# plt.show()

# Sklearn

xmin = 0
xmax = 1500
ymin = 0
ymax = 1000
# Create a structured grid to estimate kernel density
xgrid = np.arange(xmin, xmax, 5)
ygrid = np.arange(ymin, ymax, 5)
X, Y = np.meshgrid(xgrid, ygrid)
# x, y coordinates of the grid cells vertices
xy = np.vstack([X.ravel(), Y.ravel()]).T

# Define a disk within which the KDE will be perdormed
x0, y0, radius = 1000, 500, 200
r = np.sqrt((xy[:, 0] - x0)**2 + (xy[:, 1] - y0)**2)
inside = r < radius
xyu = xy[inside]  # Create mask

bw = 1.618
kde = KernelDensity(kernel='gaussian',  # Fit kernel density
                    bandwidth=bw).fit(xykde)
score = np.exp(kde.score_samples(xyu))  # Sample at the desired grid cells

score -= score.min()  # Normalize
score /= score.max()

score += 1
score = score**-1
score -= score.min()
score /= score.max()

# Assign the computed scores to the gric
z = np.full(inside.shape, 1, dtype=float)  # Create array fileld with 1
z[inside] = score
z = np.flipud(z.reshape(shape[1], shape[2]))  # Flip to correspond

plt.imshow(z,
           vmin=0,
           vmax=1,
           extent=(xmin, xmax, ymin, ymax),
           cmap='RdGy')
plt.colorbar()
plt.contour(mp.x, mp.y, h_true_obs.reshape(shape[1], shape[2]), [0], colors='red')
wells_xy = np.load(jp(cwd, 'grid', 'iw.npy'), allow_pickle=True)[:, :2]
plt.plot(1000, 500, 'wo', alpha=.7)
plt.plot(wells_xy[:, 0], wells_xy[:, 1], 'co', alpha=.7, markersize=7, markeredgecolor='w', markeredgewidth=.5)
plt.grid(color='w', linestyle='-', linewidth=.5, alpha=.2)
plt.xlim(800, 1200)
plt.ylim(300, 700)
plt.savefig(jp(bel_dir, '{}comp.png'.format(sample_n)), dpi=300)
plt.show()
