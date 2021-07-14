#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import matplotlib.pyplot as plt
import numpy as np
from skbel.algorithms import matrix_paste
from skbel.goggles import _proxy_legend
from skbel.spatial import blocks_from_rc, get_centroids

from bel4ed.config import Setup
from bel4ed.goggles._visualization import plot_wells

mt_icbund_file = jp(
    "/Users/robin/PycharmProjects/BEL/bel4ed/spatial/parameters/mt3d_icbund.npy"
)
icbund = np.load(mt_icbund_file)

delc = np.load("/Users/robin/PycharmProjects/BEL/bel4ed/spatial/parameters/delc.npy")
delr = np.load("/Users/robin/PycharmProjects/BEL/bel4ed/spatial/parameters/delr.npy")
hey = blocks_from_rc(delr, delc)
xy = np.mean(hey, axis=1)

# Check what we've done: plot the active zone.
val_icbund = [item for sublist in icbund[0] for item in sublist]  # Flattening
# Define a dummy grid with large cell dimensions to plot on it.
grf_dummy = 10
nrow_dummy = int(np.round(np.cumsum(delr)[-1]) / grf_dummy)
ncol_dummy = int(np.round(np.cumsum(delc)[-1]) / grf_dummy)
array_dummy = np.zeros((nrow_dummy, ncol_dummy))
xy_dummy = get_centroids(array_dummy, grf=grf_dummy)

inds = matrix_paste(xy_dummy, xy)
val_dummy = [val_icbund[k] for k in inds]  # Contains k values for refined grid
# Reshape in n layers x n cells in refined grid.
val_dummy_r = np.reshape(val_dummy, (nrow_dummy, ncol_dummy))

grid_dim = Setup.GridDimensions
extent = (grid_dim.xo, grid_dim.x_lim, grid_dim.yo, grid_dim.y_lim)
wells = Setup.Wells

plt.imshow(val_dummy_r, extent=extent, cmap="coolwarm")
plt.title(
    f"Transport simulation. Total cells: {np.product(icbund.shape)}. Active cells: {np.count_nonzero(icbund)}"
)
plt.xlabel("X(m)", fontsize=11)
plt.ylabel("Y(m)", fontsize=11)
plot_wells(wells, markersize=3.5)
labels = ["Inactive", "Active"]
colors = ["blue", "red"]
_proxy_legend(colors=colors, marker=["o", "o"], labels=labels, loc=2)
plt.savefig("transport_active.png", bbox_inches="tight", dpi=300, transparent=False)

plt.show()
