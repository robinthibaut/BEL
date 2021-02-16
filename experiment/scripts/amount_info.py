#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os

import numpy as np
from typing import List

from experiment._core import setup

from experiment._visualization import mode_histo

Root = List[str]


def by_mode(root: Root):
    """
    Computes the combined amount of information for n observations.
    see also
    https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
    :param root: list: List containing the roots whose wells contributions will be taken into account.
    :return:
    """

    if not isinstance(root, (list, tuple)):
        root: list = [root]

    # Deals with the fact that only one root might be selected
    fig_name = 'average'
    an_i = 0  # Annotation index
    if len(root) == 1:
        fig_name = root[0]
        an_i = 2

    wid = list(map(str, setup.wells.combination))  # Wel identifiers (n)
    wm = np.zeros((len(wid), setup.forecast.n_posts))  # Summed MHD when well #i appears
    colors = setup.wells.colors

    for r in root:  # For each root
        droot = os.path.join(setup.directories.forecasts_dir, r)  # Starting point = root folder in forecast directory
        for e in wid:  # For each subfolder (well) in the main folder
            fmhd = os.path.join(droot, e, 'obj', 'haus.npy')  # Get the MHD file
            mhd = np.load(fmhd)  # Load MHD
            idw = int(e) - 1  # -1 to respect 0 index (Well index)
            wm[idw] += mhd  # Add MHD at each well

    mode_histo(colors=colors, an_i=an_i, wm=wm, fig_name=fig_name)


if __name__ == '__main__':
    # Amount of information
    forecast_dir = setup.directories.forecasts_dir
    listit = os.listdir(forecast_dir)
    listit.remove('base')
    duq = list(filter(lambda f: os.path.isdir(os.path.join(forecast_dir, f)), listit))  # Folders of combinations

    by_mode(duq)
    by_mode(['818bf1676c424f76b83bd777ae588a1d'])
