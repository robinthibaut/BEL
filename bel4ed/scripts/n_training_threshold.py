#  Copyright (c) 2021. Robin Thibaut, Ghent University
import numpy as np
from os.path import join as jp

from bel4ed.config import Setup

sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 750, 1000, 2000]
folders = [f"n_thr_{i}_1_6492" for i in sizes]
md = Setup.Directories.forecasts_dir
obj = "uq_modified_hausdorff.npy"

files = [jp(md, f, obj) for f in folders]

uq = np.array([np.load(fi) for fi in files])

np.mean(uq, axis=2)

