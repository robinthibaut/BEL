#  Copyright (c) 2021. Robin Thibaut, Ghent University

import string
from os.path import join as jp

import joblib
from loguru import logger
from matplotlib import pyplot as plt
from skbel.goggles import _my_alphabet

import bel4ed.goggles as myvis
from bel4ed.config import Setup
from bel4ed.datasets import i_am_root, load_dataset

# COLOR = 'w'
# plt.rcParams['text.color'] = COLOR
# plt.rcParams['axes.labelcolor'] = COLOR
# plt.rcParams['xtick.color'] = COLOR
# plt.rcParams['ytick.color'] = COLOR

if __name__ == "__main__":

    training_file = jp(Setup.Directories.storage_dir, "training_roots_aniso_400.dat")
    test_file = jp(Setup.Directories.storage_dir, "test_roots_aniso_100.dat")
    training_r, test_r = i_am_root(training_file=training_file, test_file=test_file)

    # roots = test_r
    #
    # roots = ["818bf1676c424f76b83bd777ae588a1d",
    #          "fa4e291de56b4604b8cb53943c18e5e2",
    #          "d08b2a80c7ef431b92ae3c6fd9cb6482",
    #          "818dd775ba0c4e8aa567b5d7d153c9db"]

    roots = ["818bf1676c424f76b83bd777ae588a1d"]

    X, Y = load_dataset(subdir="data_structural")
    X_train = X.loc[training_r]
    X_test = X.loc[test_r]
    y_train = Y.loc[training_r]
    y_test = Y.loc[test_r]

    alphabet = string.ascii_uppercase

    fc = Setup.Focus()
    x_lim, y_lim, grf = fc.x_range, fc.y_range, fc.cell_dim

    # ['123456', '1', '2', '3', '4', '5', '6']

    base_dir = jp(Setup.Directories.forecasts_dir, "aniso_400_100_1757085")

    for i, sample in enumerate(test_r):
        logger.info(f"Plotting root {sample}")

        wells = ["123456"]

        for j, w in enumerate(wells):

            logger.info(f"Plotting well {w}")

            if w == "123456":
                annotation = _my_alphabet(i)
            else:
                annotation = _my_alphabet(j - 1)

            # BEL pickle
            md = jp(base_dir, sample, w)
            bel = joblib.load(jp(md, "obj", "bel.pkl"))
            bel.Y_shape = (1, 100, 100)

            logger.info(f"Plotting results")
            myvis.plot_results(
                bel,
                X=X_train,
                X_obs=X_test.iloc[i],
                Y=y_train,
                Y_obs=y_test.iloc[i],
                base_dir=base_dir,
                root=sample,
                folder=w,
                annotation=annotation,
            )
            plt.close()

            logger.info(f"Plotting PCA")
            myvis.pca_vision(
                bel,
                X_obs=X_test.iloc[i],
                Y_obs=y_test.iloc[i],
                base_dir=base_dir,
                w=w,
                root=sample,
                d=True,
                h=True,
                exvar=False,
                before_after=True,
                labels=True,
                scores=True,
            )
            plt.close()

        logger.info(f"Plotting K")
        myvis.plot_K_field(base_dir=base_dir, root=sample)
        plt.close()

        logger.info(f"Plotting HEAD")
        myvis.plot_head_field(base_dir=base_dir, root=sample)
        plt.close()

        # logger.info(f"Plotting CCA")
        myvis.cca_vision(
            base_dir=base_dir,
            Y_obs=y_test.iloc[i].to_numpy().reshape(1, -1),
            root=sample,
            folders=wells,
        )
