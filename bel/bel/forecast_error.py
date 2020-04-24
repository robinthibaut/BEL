#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

from bel.processing.signed_distance import SignedDistance
from bel.toolbox.hausdorff import modified_distance
from bel.toolbox.plots import Plot, cca_plot
from bel.toolbox.posterior_ops import PosteriorOps

# plt.style.use('dark_background')
plt.style.use('cyberpunk')

po = PosteriorOps()
x_lim, y_lim, grf = [800, 1150], [300, 700], 2
mp = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf)

# Directories & files paths
cwd = os.getcwd()
wdir = jp('bel', 'hydro', 'grid')
mp.wdir = wdir
study_folder = '5c254d8f-7ab6-4627-b3a3-7a50564e56b7'
bel_dir = jp('bel', 'forecasts', study_folder)
res_dir = jp(bel_dir, 'objects')
fig_dir = jp(bel_dir, 'figures')
fig_cca_dir = jp(fig_dir, 'CCA')
fig_pred_dir = jp(fig_dir, 'Predictions')

# Load objects
f_names = list(map(lambda fn: jp(res_dir, fn + '.pkl'), ['cca', 'd_pca', 'h_pca']))
cca_operator, d_pco, h_pco = list(map(joblib.load, f_names))

# Inspect transformation between physical and PC space
dnc0 = d_pco.ncomp
hnc0 = h_pco.ncomp
print(d_pco.perc_pca_components(dnc0))
print(h_pco.perc_pca_components(hnc0))
mp.pca_inverse_compare(d_pco, h_pco, dnc0, hnc0)

# Cut desired number of PC components
d_pc_training, d_pc_prediction = d_pco.pca_refresh(dnc0)
h_pc_training, h_pc_prediction = h_pco.pca_refresh(hnc0)

# CCA plots
d_cca_training, h_cca_training = cca_operator.transform(d_pc_training, h_pc_training)
d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T
cca_plot(cca_operator, d_cca_training, h_cca_training, d_pc_prediction, h_pc_prediction, sdir=fig_cca_dir)

# %% Random sample from the posterior
sample_n = 0
n_posts = 500
forecast_posterior = po.random_sample(sample_n=sample_n,
                                      pca_d=d_pco,
                                      pca_h=h_pco,
                                      cca_obj=cca_operator,
                                      n_posts=n_posts,
                                      add_comp=0)
# Get the true array of the prediction
d_pc_obs = d_pco.predict_pc[sample_n]  # Prediction set - PCA space
shape = h_pco.raw_data.shape
h_true_obs = h_pco.predict_physical[sample_n].reshape(shape[1], shape[2])  # Prediction set - physical space

# Predicting the function based for a certain number of 'observations'
h_pc_true_pred = cca_operator.predict(d_pc_obs[:d_pco.ncomp].reshape(1, -1))
# Going back to the original function dimension and reshape.
h_pred = h_pco.inverse_transform(h_pc_true_pred).reshape(shape[1], shape[2])

# Plot results
ff = jp(fig_pred_dir, '{}_{}.png'.format(sample_n, cca_operator.n_components))
mp.whp_prediction(forecasts=forecast_posterior,
                  h_true=h_true_obs,
                  h_pred=h_pred,
                  show_wells=True,
                  fig_file=ff)

# %% extract 0 contours
vertices = mp.contours_vertices(forecast_posterior)

# Reshape coordinates
x_stack = np.hstack([vi[:, 0] for vi in vertices])
y_stack = np.hstack([vi[:, 1] for vi in vertices])
# Final array np.array([[x0, y0],...[xn,yn]])
xykde = np.vstack([x_stack, y_stack]).T

# %% Kernel density

# Scatter plot vertices
nn = sample_n
plt.plot(vertices[nn][:, 0], vertices[nn][:, 1], 'o-')
plt.show()

# Grid geometry
xmin = x_lim[0]
xmax = x_lim[1]
ymin = y_lim[0]
ymax = y_lim[1]
# Create a structured grid to estimate kernel density
# TODO: create a function to copy/paste values on differently refined grids
# Prepare the Plot instance with right dimensions
grf_kd = 2
mpkde = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf_kd)
mpkde.wdir = wdir
cell_dim = grf_kd
xgrid = np.arange(xmin, xmax, cell_dim)
ygrid = np.arange(ymin, ymax, cell_dim)
X, Y = np.meshgrid(xgrid, ygrid)
# x, y coordinates of the grid cells vertices
xy = np.vstack([X.ravel(), Y.ravel()]).T

# Define a disk within which the KDE will be performed to save time
x0, y0, radius = 1000, 500, 200
r = np.sqrt((xy[:, 0] - x0) ** 2 + (xy[:, 1] - y0) ** 2)
inside = r < radius
xyu = xy[inside]  # Create mask

# Perform KDE
bw = 1.618  # Arbitrary 'smoothing' parameter
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
z = np.flipud(z.reshape(X.shape))  # Flip to correspond to actual distribution.

# Plot KDE
mp.whp(h_true_obs.reshape(1, shape[1], shape[2]),
       alpha=1,
       lw=1,
       show_wells=True,
       colors='red',
       show=False)
mpkde.whp(bkg_field_array=z,
          vmin=None,
          vmax=None,
          cmap='RdGy',
          colors='red',
          fig_file=jp(fig_pred_dir, '{}comp.png'.format(sample_n)),
          show=True)

# %% New approach : stack binary WHPA
# For this approach we use our SignedDistance module
sd_kd = SignedDistance(x_lim=x_lim, y_lim=y_lim, grf=grf)
mpbin = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf)
mpbin.wdir = wdir
bin_whpa = [sd_kd.matrix_poly_bin(pzs=p, inside=1/n_posts, outside=0) for p in vertices]
big_sum = np.sum(bin_whpa, axis=0)  # Stack them
b_low = np.where(big_sum == 0, 1, big_sum)  # Replace 0 values by 1
mpbin.whp(bkg_field_array=b_low,
          show_wells=True,
          vmin=None,
          vmax=None,
          cmap='RdGy',
          fig_file=jp(fig_pred_dir, '{}_0stacked.png'.format(sample_n)),
          show=True)

# a measure of the error could be a measure of the area covered by the n samples.
error = len(np.where(b_low < 1)[0])  # Number of cells covered at least once.


#  Let's try Hausdorff...

v_h_true = mp.contours_vertices(h_true_obs)[0]
v_h_pred = mp.contours_vertices(h_pred)[0]
mhd = modified_distance(v_h_true, v_h_pred)

mhds = np.array([modified_distance(v_h_true, vt) for vt in vertices])
print(mhds.mean())
print(mhds.min())
print(mhds.max())

min_pos = np.where(mhds == mhds.min())[0][0]
max_pos = np.where(mhds == mhds.max())[0][0]


# Plot results
ff = jp(fig_pred_dir, '{}_{}_hausdorff.png'.format(sample_n, cca_operator.n_components))
mp.whp_prediction(forecasts=np.expand_dims(forecast_posterior[max_pos], axis=0),
                  h_true=h_true_obs,
                  h_pred=forecast_posterior[min_pos],
                  show_wells=True,
                  fig_file=ff)
