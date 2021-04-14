#  Copyright (c) 2021. Robin Thibaut, Ghent University
import numpy as np
from os.path import join as jp
import matplotlib.pyplot as plt

from bel4ed.config import Setup

sizes = [125, 150, 200, 250, 400, 500, 750, 1000, 2000]
folders = [f"n_thr_{i}_1_6492" for i in sizes]
md = Setup.Directories.forecasts_dir
obj = "uq_modified_hausdorff.npy"

files = [jp(md, f, obj) for f in folders]

uq = [np.load(fi) for fi in files]

# means = [np.mean(e)/e.shape[1] for e in uq]
means = [np.mean(e) for e in uq]
means = np.array(list(zip(sizes, means)))

print(means)

plt.plot(means[:, 0], means[:, 1], "-b")
plt.plot(means[:, 0], means[:, 1], "ko")
plt.title("Figure 1")
plt.xlabel("Training size")
plt.ylabel("Average error")
plt.show()
