#  Copyright (c) 2021. Robin Thibaut, Ghent University
import os
from os.path import join as jp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skbel.spatial import grid_parameters

from bel4ed import Setup
from bel4ed.datasets import data_loader
from bel4ed.goggles._visualization import (curves_i, plot_head_field,
                                           plot_K_field, whpa_plot)
from bel4ed.preprocessing import beautiful_curves, signed_distance
from bel4ed.utils import i_am_framed

p = "/Users/robin/Desktop/samples"

listme = os.listdir(p)
# Filter folders out
folders = list(filter(lambda fo: os.path.isdir(os.path.join(p, fo)), listme))

root = "1d1e18ec7b954ef886fc5e14487b704b"

# tc0, sd, r = data_loader(
#     res_dir=p,
#     roots=[root],
#     d=True,
#     h=True,
# )

# %% Predictor
# Refined breakthrough curves data file
n_time_steps = Setup.HyperParameters.n_tstp
# Loads the results:
# tc has shape (n_sim, n_wells, n_time_steps)
predictor = pd.DataFrame()
for i, f in enumerate(folders):
    tc_training = beautiful_curves(
        res_dir=p,
        ids=[f],
        n_time_steps=n_time_steps,
    )

    ids = {"root": f}
    # ids = {"root": f, "id": [f"well_{i}" for i in range(tc_training.shape[1])]}

    df_predictor = i_am_framed(array=tc_training[0], ids=ids, flat=True)
    predictor = predictor.append(df_predictor)

    df_predictor.to_pickle(jp(p, f, f"predictor.pkl"))

file_name = jp(p, "predictor.pkl")
shape = df_predictor.attrs["physical_shape"]
# Get original shape
predictor.attrs = df_predictor.attrs
# Set columns names corresponding to source number (curves are flattened)
columns = np.concatenate(
    [np.ones(shape[1], dtype=int) * i for i in range(1, shape[0] + 1)]
)
predictor.columns = list(map(str, columns))
predictor.to_pickle(file_name)

# %% Target
target = pd.DataFrame()
for i, f in enumerate(folders):
    x_lim, y_lim, grf = Setup.Focus.x_range, Setup.Focus.y_range, Setup.Focus.cell_dim

    _, pzs_training, _ = data_loader(res_dir=p, roots=[f], h=True)

    # Load parameters:
    xys, nrow, ncol = grid_parameters(
        x_lim=x_lim, y_lim=y_lim, grf=grf
    )  # Initiate SD instance

    # Compute signed distance on pzs.
    # h is the matrix of target feature on which PCA will be performed.
    h_training = np.array(
        [signed_distance(xys, nrow, ncol, grf, pp) for pp in pzs_training]
    )

    ids = {"root": f}
    # Convert to dataframes
    df_target = i_am_framed(array=h_training, ids=ids)

    target = target.append(df_target)

    df_target.to_pickle(jp(p, f, f"target.pkl"))

file_name = jp(p, "target.pkl")
target.attrs = df_target.attrs
target.to_pickle(file_name)

wells = Setup.Wells
wells_id = list(wells.wells_data.keys())
cols = [wells.wells_data[w]["color"] for w in wells_id if "pumping" not in w]

tc = predictor.to_numpy().reshape((-1, 6, 200))

curves_i(cols=cols, tc=tc, show=True)

for f in folders:
    plot_K_field(k_dir=p, base_dir=p, root=f, show=True)

# plot_head_field(base_dir=p, root=root)

fig_dir = jp(p, root)
ff = jp(fig_dir, f"{root}.pdf")  # figure name
h_training = target.to_numpy().reshape((-1,) + (100, 100))
# Plots target training + prediction
for hg in h_training:
    whpa_plot(whpa=hg, color="blue", alpha=1, x_lim=[750, 1150], show=True)
