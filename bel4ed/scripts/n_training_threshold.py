#  Copyright (c) 2021. Robin Thibaut, Ghent University
import numpy as np
from os.path import join as jp
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bel4ed.config import Setup

sizes = np.concatenate([range(125, 500, 25), range(500, 1000, 100)])

sizes = np.concatenate([[125], range(150, 500, 50), range(500, 1000, 100)])

rand = [
    287071,
    437176,
    446122,
    502726,
    656184,
    791302,
    824883,
    851117,
    885166,
    980285,
    15843,
    202157,
    235506,
    430849,
    547976,
    617924,
    862286,
    863668,
    975934,
    993935
]

fig, ax = plt.subplots()

for rs in rand:
    folders = [f"n_thr_{i}_1_{rs}" for i in sizes]
    md = Setup.Directories.forecasts_dir
    obj = "uq_structural_similarity.npy"

    files = [jp(md, f, obj) for f in folders]

    uq = [np.load(fi) for fi in files]

    # means = [np.mean(e)/e.shape[1] for e in uq]
    means = [-np.mean(e) for e in uq]
    means = np.array(list(zip(sizes, means)))

    # print(means)

    ax.plot(means[:, 0], means[:, 1], "-", alpha=.8, linewidth=2)
    # ax.plot(means[:, 0], means[:, 1], "lightblue", "o", markersize=1)
    # plt.title("")
plt.axvline(x=400)
plt.grid(alpha=0.3)
plt.xlim([125, 900])
# plt.ylim([-.822, -.667])
plt.xlabel("Training size")
plt.ylabel("Average SSIM index")
plt.savefig("n_training.pdf", dpi=300, bbox_inches="tight", transparent=True)
plt.show()
