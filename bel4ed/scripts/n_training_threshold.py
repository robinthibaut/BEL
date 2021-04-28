#  Copyright (c) 2021. Robin Thibaut, Ghent University
import numpy as np
from os.path import join as jp
import matplotlib.pyplot as plt

from bel4ed.config import Setup

sizes = [125, 150, 200, 250, 400, 500, 750, 1000, 2000]
rs = ["287071", "437176",
      "446122", "502726",
      "656184", "791302",
      "824883", "851117",
      "885166", "980285"]

fig, ax = plt.subplots()

for rand in rs:
    folders = [f"n_thr_{i}_1_{rand}" for i in sizes]
    md = Setup.Directories.forecasts_dir
    obj = "uq_modified_hausdorff.npy"

    files = [jp(md, f, obj) for f in folders]

    uq = [np.load(fi) for fi in files]

    # means = [np.mean(e)/e.shape[1] for e in uq]
    means = [np.mean(e) for e in uq]
    means = np.array(list(zip(sizes, means)))

    # print(means)

    ax.plot(means[:, 0], means[:, 1], "-")
    # ax.plot(means[:, 0], means[:, 1], "+")
    # plt.title("")
plt.grid(alpha=0.3)
plt.ylim([6.5, 15.2])
plt.xlabel("Training size")
plt.ylabel("Average error")
plt.show()
