#  Copyright (c) 2021. Robin Thibaut, Ghent University

from os.path import join as jp
import multiprocessing as mp

import joblib
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from skbel.goggles import _my_alphabet

import bel4ed.goggles as myvis
from bel4ed.config import Setup
from bel4ed.datasets import i_am_root, load_dataset

base_dir = jp(Setup.Directories.forecasts_dir, "aniso_1000_250_3529748")

training_file = jp(Setup.Directories.storage_dir, "training_roots_aniso_1000_3529748.dat")
test_file = jp(Setup.Directories.storage_dir, "test_roots_aniso_250_3529748.dat")
training_r, test_r = i_am_root(training_file=training_file, test_file=test_file)

# root = "105e7b81e65b46c1bbb096035139ca01"
root = "3c6003846f7449cd8715ee7a8363e2aa"
wells = ["123456"]

X, Y = load_dataset(subdir="data_structural")
X_train = X.loc[training_r]
X_test = X.loc[test_r]
y_train = Y.loc[training_r]
y_test = Y.loc[test_r]


def plotter(sample):
    logger.info(f"Plotting root {sample}")

    logger.info(f"Plotting K")
    myvis.plot_K_field(base_dir=base_dir, root=sample, annotation=kan)
    plt.close()

    for j, w in enumerate(wells):

        logger.info(f"Plotting well {w}")

        if w == "123456":
            annotation = _my_alphabet(j)
        elif j > 0:
            annotation = _my_alphabet(j - 1)
        else:
            annotation = _my_alphabet(j)

        annotation = nan

        md = jp(base_dir, sample, w)
        bel = joblib.load(jp(md, "obj", "bel.pkl"))

        logger.info(f"Plotting results")
        myvis.plot_results(
            bel,
            X=X_train,
            X_obs=X_test.loc[sample],
            Y=y_train,
            Y_obs=y_test.loc[sample],
            base_dir=base_dir,
            root=sample,
            folder=w,
            annotation=annotation,
        )
        plt.close()

        # logger.info(f"Plotting PCA")
        # myvis.pca_vision(
        #     bel,
        #     X_obs=X_test.iloc[i],
        #     Y_obs=y_test.iloc[i],
        #     base_dir=base_dir,
        #     w=w,
        #     root=sample,
        #     d=True,
        #     h=True,
        #     exvar=False,
        #     before_after=True,
        #     labels=True,
        #     scores=True,
        # )
        # plt.close()

    # logger.info(f"Plotting HEAD")
    # myvis.plot_head_field(base_dir=base_dir, root=sample)
    # plt.close()

    # logger.info(f"Plotting CCA")
    # myvis.cca_vision(
    #     base_dir=base_dir,
    #     Y_obs=y_test.iloc[i].to_numpy().reshape(1, -1),
    #     root=sample,
    #     folders=wells,
    # )
    # plt.close()


if __name__ == "__main__":

    samples = ["d47eafc2601c4b95a2eb06faaa2fb9df",
               "ce4255c17dcf47cfa6713ad8b0d15cbd",
               "b8ffc46efedf4b5981d58b59a237e1d8",
               "57b5cbb8b8014b3fad1b75bcdb8e3601",
               "3a50d88a4f7c4b7eb69b29682b2ed410",
               "bc3b2b0b3bc0431da5aac2318756f64e"]

    an = np.arange(0, 12, 2)
    ak = np.arange(1, 13, 2)
    for i, s in enumerate(samples):
        kan = _my_alphabet(ak[i])
        nan = _my_alphabet(an[i])
        plotter(s)

    # plotter(root)
    # n_cpu = 10
    # pool = mp.Pool(n_cpu)
    # pool.map(plotter, test_r)
    # pool.close()
    # pool.join()

# d47eafc2601c4b95a2eb06faaa2fb9df
# ce4255c17dcf47cfa6713ad8b0d15cbd
# b8ffc46efedf4b5981d58b59a237e1d8
# 57b5cbb8b8014b3fad1b75bcdb8e3601
# 3a50d88a4f7c4b7eb69b29682b2ed410
# bc3b2b0b3bc0431da5aac2318756f64e
