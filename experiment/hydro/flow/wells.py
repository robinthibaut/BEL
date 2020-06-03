#  Copyright (c) 2020. Robin Thibaut, Ghent University

from os.path import join as jp

import numpy as np
import pandas as pd
from pysgems.io.sgio import export_eas

from experiment.base.inventory import Directories, Wels


# Define injection wells
# [ [X, Y, [r#1, r#2... r#n]] ]
# Pumping well located at [1000, 500]
# Template
# wells_data = np.array([[980, 500, [0, 24, 0]]])


def gen_rand_well(radius, x0, y0):
    """
    Generates random coordinates within a circle.
    :param radius:
    :param x0:
    :param y0:
    :return:
    """
    # random angle
    alpha = 2 * np.pi * np.random.random()
    # random radius
    r = 50 + radius * np.sqrt(np.random.random())
    # calculating coordinates
    x = r * np.cos(alpha) + x0
    y = r * np.sin(alpha) + y0

    return [x, y]


def well_maker():
    # Directories
    md = Directories()
    grid_dir = md.grid_dir
    wd = Wels().wels_data

    columns = ['x', 'y', 'hd']  # Save wells data for sgems
    wels_xy = [wd[o]['coordinates'] for o in wd]
    wels_val = np.ones((len(wels_xy), 1)) * -9966699
    wel_arr = np.concatenate((wels_xy, wels_val), axis=1)
    df = pd.DataFrame(columns=columns, data=wel_arr)
    export_eas(df, jp(grid_dir, 'wels'))


if __name__ == '__main__':
    well_maker()
