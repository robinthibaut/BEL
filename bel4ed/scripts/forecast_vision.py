#  Copyright (c) 2021. Robin Thibaut, Ghent University

import string
import cProfile
from os.path import join as jp
import multiprocessing as mp

import joblib
from loguru import logger
from matplotlib import pyplot as plt
from skbel.goggles import _my_alphabet

import bel4ed.goggles as myvis
from bel4ed.config import Setup
from bel4ed.datasets import i_am_root, load_dataset

base_dir = jp(Setup.Directories.forecasts_dir, "c23_1000_250_8872963")

training_file = jp(
    Setup.Directories.storage_dir, "training_roots_c23_1000_8872963.dat"
)
test_file = jp(Setup.Directories.storage_dir, "test_roots_c23_250_8872963.dat")
training_r, test_r = i_am_root(training_file=training_file, test_file=test_file)

root = "42b239a3ad0441e6bafe5f6b61daa84c"
wells = ["26"]

X, Y = load_dataset()
X_train = X.loc[training_r]
X_test = X.loc[test_r]
y_train = Y.loc[training_r]
y_test = Y.loc[test_r]


def plotter(sample):
    logger.info(f"Plotting root {sample}")

    for j, w in enumerate(wells):

        logger.info(f"Plotting well {w}")

        if w == "123456":
            annotation = _my_alphabet(j)
        elif j > 0:
            annotation = _my_alphabet(j - 1)
        else:
            annotation = _my_alphabet(j)

        annotation = "B"
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

        logger.info(f"Plotting PCA")
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

    logger.info(f"Plotting K")
    myvis.plot_K_field(base_dir=base_dir, root=sample)
    plt.close()

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
    plotter(root)
    # cProfile.run("plotter('0ab8c0ae73bf481bac8e9310b2a7af6d')")
    # n_cpu = 10
    # pool = mp.Pool(n_cpu)
    # pool.map(plotter, test_r)
    # pool.close()
    # pool.join()
