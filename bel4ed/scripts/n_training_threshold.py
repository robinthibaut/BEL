#  Copyright (c) 2021. Robin Thibaut, Ghent University
import numpy as np
from os.path import join as jp
import matplotlib.pyplot as plt

from bel4ed.config import Setup

sizes = range(125, 500, 25)

rand = [287071, 437176,
        446122, 502726,
        656184, 791302,
        824883, 851117,
        885166, 980285,
        15843, 202157,
        235506, 430849,
        547976, 617924,
        862286, 863668,
        975934, 993935,
        1561, 1998,
        1678941, 19619,
        125691, 168994652,
        16516, 5747,
        156886, 218766,
        21518, 51681,
        6546844, 5418717]

fig, ax = plt.subplots()

for rs in rand:
    folders = [f"n_thr_{i}_1_{rs}" for i in sizes]
    md = Setup.Directories.forecasts_dir
    obj = "uq_structural_similarity.npy"

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
# plt.xlim([125, 500])
# plt.ylim([6.5, 15.2])
plt.xlabel("Training size")
plt.ylabel("Average error")
plt.savefig("n_training.png")
plt.show()
