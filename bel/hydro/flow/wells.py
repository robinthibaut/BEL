#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import numpy as np
import pandas as pd
from pysgems.io.sgio import export_eas


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
    mod_dir = os.getcwd()  # Module directory
    grid_dir = jp(os.path.dirname(mod_dir), 'grid')
    # Pumping well data
    pumping_well = np.array([[1000, 500, [-1000, -1000, -1000]]])
    pw_xy = pumping_well[0, :2]

    # Injection wells location
    iw_xy = [pw_xy + [-50, -50],
             pw_xy + [-70, 60],
             pw_xy + [-100, 5],
             pw_xy + [68, 15],
             pw_xy + [30, 80],
             pw_xy + [50, -30]]

    wells_data = np.array([[wxy[0], wxy[1], [0, 24, 0]] for wxy in iw_xy])

    np.save(jp(grid_dir, 'iw'), wells_data)  # Saves injection wells stress period data
    np.save(jp(grid_dir, 'pw'), pumping_well)  #

    columns = ['x', 'y', 'hd']  # Save wells data for sgems
    wels_xy = np.concatenate(([pw_xy], iw_xy), axis=0)
    wels_val = np.ones((len(wels_xy), 1)) * -9966699
    wel_arr = np.concatenate((wels_xy, wels_val), axis=1)
    df = pd.DataFrame(columns=columns, data=wel_arr)
    export_eas(df, jp(grid_dir, 'wels'))


if __name__ == '__main__':
    well_maker()
