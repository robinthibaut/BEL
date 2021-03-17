#  Copyright (c) 2021. Robin Thibaut, Ghent University
import os
from os.path import join as jp

import pandas as pd

from bel4ed.config import Setup
from bel4ed.utils import i_am_framed
from bel4ed.preprocessing import beautiful_curves

md = Setup.Directories.hydro_res_dir
listme = os.listdir(md)

# Filter folders out
folders = list(filter(lambda f: os.path.isdir(os.path.join(md, f)), listme))

# Refined breakthrough curves data file
n_time_steps = Setup.HyperParameters.n_tstp
# Loads the results:
# tc has shape (n_sim, n_wells, n_time_steps)
tc_training = beautiful_curves(
    res_dir=md,
    ids=[folders[0]],
    n_time_steps=n_time_steps,
)

ids = {"root": folders[0], "id": [f"well_{i}" for i in range(tc_training.shape[1])],}

df = i_am_framed(array=tc_training[0], ids=ids, flat=False)

df.to_pickle(jp(md, folders[0], f"predictor.pkl"))

