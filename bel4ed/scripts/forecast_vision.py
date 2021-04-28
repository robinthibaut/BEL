#  Copyright (c) 2021. Robin Thibaut, Ghent University

import string
from os.path import join as jp

import joblib
from loguru import logger

import bel4ed.goggles as myvis
from bel4ed.config import Setup
from bel4ed.datasets import i_am_root

# COLOR = 'w'
# plt.rcParams['text.color'] = COLOR
# plt.rcParams['axes.labelcolor'] = COLOR
# plt.rcParams['xtick.color'] = COLOR
# plt.rcParams['ytick.color'] = COLOR

if __name__ == "__main__":

    training_file = jp(Setup.Directories.storage_dir, "training_roots_400.dat")
    test_file = jp(Setup.Directories.storage_dir, "test_roots_100.dat")
    training_r, test_r = i_am_root(training_file=training_file, test_file=test_file)

    # roots = test_r

    roots = ["818bf1676c424f76b83bd777ae588a1d"]

    alphabet = string.ascii_uppercase

    fc = Setup.Focus()
    x_lim, y_lim, grf = fc.x_range, fc.y_range, fc.cell_dim

    # ['123456', '1', '2', '3', '4', '5', '6']

    base_dir = jp(Setup.Directories.forecasts_dir, "test_400_1")

    for i, sample in enumerate(roots):
        logger.info(f"Plotting root {sample}")

        wells = ["123456"]

        for j, w in enumerate(wells):

            logger.info(f"Plotting well {w}")

            if w == "123456":
                annotation = alphabet[i]
            else:
                annotation = alphabet[j - 1]

            # BEL pickle
            md = jp(base_dir, sample, w)
            bel = joblib.load(jp(md, "obj", "bel.pkl"))

            myvis.plot_results(
                bel, base_dir=base_dir, root=sample, folder=w, annotation=annotation, d=True
            )

            myvis.pca_vision(
                bel,
                base_dir=base_dir,
                w=w,
                root=sample,
                d=True,
                h=True,
                exvar=True,
                before_after=True,
                labels=True,
                scores=True,
            )

        myvis.plot_K_field(base_dir=base_dir, root=sample)

        myvis.plot_head_field(base_dir=base_dir, root=sample)

        myvis.cca_vision(base_dir=base_dir, root=sample, folders=wells)
