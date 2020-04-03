import os
from os.path import join as jp

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

from bel.toolbox.tools import Plot, MeshOps, PosteriorOps

plt.style.use('dark_background')

mp = Plot()
mo = MeshOps()
po = PosteriorOps()

# Directories & files paths
cwd = os.getcwd()
wdir = jp('..', 'hydro', 'grid')
bel_dir = jp(cwd, 'forecasts', '370f6fac-27c4-4277-a114-5c1d70a5bdb2')
res_dir = jp(bel_dir, 'objects')
fig_dir = jp(bel_dir, 'figures')
fig_pred_dir = jp(fig_dir, 'Predictions')

# Load objects
f_names = list(map(lambda fn: jp(res_dir, fn + '.pkl'), ['cca', 'd_pca', 'h_pca']))
cca, d_pco, h_pco = list(map(joblib.load, f_names))

# Random sample from the posterior
sample_n = 4
n_posts = 500
forecast_posterior = po.random_sample(sample_n=sample_n,
                                      pca_d=d_pco,
                                      pca_h=h_pco,
                                      cca_obj=cca,
                                      n_posts=n_posts, add_comp=0)
# Get the true array of the prediction
d_pc_obs = d_pco.dpp[sample_n]
shape = h_pco.raw_data.shape
h_true_obs = h_pco.dp[sample_n].reshape(shape[1], shape[2])

# Predicting the SD based for a certain number of 'observations'
h_pc_true_pred = cca.predict(d_pc_obs[:d_pco.ncomp].reshape(1, -1))
# Going back to the original SD dimension and reshape.
h_pred = h_pco.inverse_transform(h_pc_true_pred).reshape(shape[1], shape[2])

# Plot results
ff = jp(fig_pred_dir, '{}_{}.png'.format(sample_n, cca.n_components))
mp.whp_prediction(forecasts=forecast_posterior,
                  h_true=h_true_obs,
                  h_pred=h_pred,
                  fig_file=ff)
# %% extract 0 contours

c0s = [plt.contour(mp.x, mp.y, f, [0]) for f in forecast_posterior]
plt.close()
v = np.array([c0.allsegs[0][0] for c0 in c0s])

x = np.hstack([vi[:, 0] for vi in v])
y = np.hstack([vi[:, 1] for vi in v])

xykde = np.vstack([x, y]).T

# %% Kernel density

# Scatter plot vertices
# nn = sample_n
# plt.plot(v[nn][:, 0], v[nn][:, 1], 'o-')
# plt.show()

# Grid geometry
xmin = 0
xmax = 1500
ymin = 0
ymax = 1000
# Create a structured grid to estimate kernel density
cell_dim = 5
xgrid = np.arange(xmin, xmax, cell_dim)
ygrid = np.arange(ymin, ymax, cell_dim)
X, Y = np.meshgrid(xgrid, ygrid)
# x, y coordinates of the grid cells vertices
xy = np.vstack([X.ravel(), Y.ravel()]).T

# Define a disk within which the KDE will be performed
x0, y0, radius = 1000, 500, 200
r = np.sqrt((xy[:, 0] - x0) ** 2 + (xy[:, 1] - y0) ** 2)
inside = r < radius
xyu = xy[inside]  # Create mask

# Perform KDE
bw = 1.618
kde = KernelDensity(kernel='gaussian',  # Fit kernel density
                    bandwidth=bw).fit(xykde)
score = np.exp(kde.score_samples(xyu))  # Sample at the desired grid cells


def score_norm(sc, max_score=None):
    """
    Normalizes the KDE scores.
    """
    sc -= sc.min()
    sc /= sc.max()

    sc += 1
    sc = sc ** -1

    sc -= sc.min()
    sc /= sc.max()

    return sc


# Normalize
score = score_norm(score)

# Assign the computed scores to the grid
z = np.full(inside.shape, 1, dtype=float)  # Create array filled with 1
z[inside] = score
z = np.flipud(z.reshape(shape[1], shape[2]))  # Flip to correspond

# Plot KDE
mp.whp(h_true_obs.reshape(1, shape[1], shape[2]),
       alpha=1,
       lw=1,
       bkg_field_array=z,
       vmin=None,
       vmax=None,
       cmap='RdGy',
       colors='red',
       fig_file=jp(fig_pred_dir, '{}comp.png'.format(sample_n)),
       show=True)
