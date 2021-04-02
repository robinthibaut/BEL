#  Copyright (c) 2021. Robin Thibaut, Ghent University

import string
from os.path import join as jp

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

    training_file = jp(Setup.Directories.storage_dir, "roots.dat")
    test_file = jp(Setup.Directories.storage_dir, "test_roots.dat")
    training_r, test_r = i_am_root(training_file=training_file, test_file=test_file)

    # roots = samples

    # roots = ['818bf1676c424f76b83bd777ae588a1d',
    #          'dc996e54728b4bb4a7234ee691729076',
    #          '27ec76adab2e406794584fc993188c24',
    #          '9a389395bfbe4cd883dfa3e452752978']

    roots = ["818bf1676c424f76b83bd777ae588a1d"]

    alphabet = string.ascii_uppercase

    fc = Setup.Focus()
    x_lim, y_lim, grf = fc.x_range, fc.y_range, fc.cell_dim

    # ['123456', '1', '2', '3', '4', '5', '6']

    for i, sample in enumerate(roots):
        logger.info(f"Plotting root {sample}")

        wells = ["123456", "1", "2", "3", "4", "5", "6"]

        # for j, w in enumerate(wells):
        #
        #     logger.info(f"Plotting well {w}")
        #
        #     if w == "123456":
        #         annotation = alphabet[i]
        #     else:
        #         annotation = alphabet[j - 1]
        #
        #     # BEL pickle
        #     md = jp(Setup.Directories.forecasts_dir, sample, w)
        #     bel = joblib.load(jp(md, "obj", "bel.pkl"))
        #
        #     myvis.plot_results(
        #         bel, root=sample, folder=w, annotation=annotation, d=True
        #     )
        #
        myvis.pca_vision(
            bel,
            w=w,
            root=sample,
            d=True,
            h=True,
            exvar=True,
            before_after=True,
            labels=True,
            scores=True,
        )

        # myvis.plot_K_field(root=sample)

        # myvis.plot_head_field(root=sample)
        #
        myvis.cca_vision(root=sample, folders=wells)
