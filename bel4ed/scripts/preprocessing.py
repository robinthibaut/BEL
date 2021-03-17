#  Copyright (c) 2021. Robin Thibaut, Ghent University
import os
from os.path import join as jp

import numpy as np
import pandas as pd
from loguru import logger

from bel4ed.config import Setup
from bel4ed.datasets import data_loader
from bel4ed.spatial import grid_parameters
from bel4ed.utils import i_am_framed
from bel4ed.preprocessing import beautiful_curves, signed_distance

md = Setup.Directories.hydro_res_dir
listme = os.listdir(md)

# Filter folders out
folders = list(filter(lambda fo: os.path.isdir(os.path.join(md, fo)), listme))

# %% Predictor
# Refined breakthrough curves data file
n_time_steps = Setup.HyperParameters.n_tstp
# Loads the results:
# tc has shape (n_sim, n_wells, n_time_steps)
logger.info(f"Preprocessing predictor for {len(folders)} samples")
predictor = pd.DataFrame()
for i, f in enumerate(folders):
    logger.info(f"[{i+1}/{len(folders)}] | {f}")
    tc_training = beautiful_curves(
        res_dir=md,
        ids=[f],
        n_time_steps=n_time_steps,
    )

    ids = {"root": f}
    # ids = {"root": f, "id": [f"well_{i}" for i in range(tc_training.shape[1])]}

    df_predictor = i_am_framed(array=tc_training[0], ids=ids, flat=True)
    predictor = predictor.append(df_predictor)

    df_predictor.to_pickle(jp(md, f, f"predictor.pkl"))

file_name = jp(Setup.Directories.data_dir, "data", "predictor.pkl")
logger.info(f"Saving full predictor to {file_name}")
shape = df_predictor.attrs["physical_shape"]
# Get original shape
predictor.attrs = df_predictor.attrs
# Set columns names corresponding to source number (curves are flattened)
columns = np.concatenate([np.ones(shape[1])*i for i in range(1, shape[0]+1)])
predictor.columns = columns
predictor.to_pickle(file_name)

# %% Target
logger.info(f"Preprocessing target for {len(folders)} samples")
target = pd.DataFrame()
for i, f in enumerate(folders):
    logger.info(f"[{i+1}/{len(folders)}] | {f}")
    x_lim, y_lim, grf = Setup.Focus.x_range, Setup.Focus.y_range, Setup.Focus.cell_dim

    _, pzs_training, _ = data_loader(roots=[f], h=True)

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

    df_target.to_pickle(jp(md, f, f"target.pkl"))

file_name = jp(Setup.Directories.data_dir, "data", "target.pkl")
logger.info(f"Saving full target to {file_name}")
target.attrs = df_target.attrs
target.to_pickle(file_name)
