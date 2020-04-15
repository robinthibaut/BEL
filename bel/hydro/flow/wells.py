import os
from os.path import join as jp
import numpy as np

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

    # Pumping well data
    pumping_well = np.array([[1000, 500, [-1000, -1000, -1000]]])
    pw_xy = pumping_well[0, :2]

    # Injection wells location
    wells_xy = [pw_xy + [-50, -50],
                pw_xy + [-70, 60],
                pw_xy + [-100, 5],
                pw_xy + [68, 15],
                pw_xy + [30, 80],
                pw_xy + [50, -30]]

    wells_data = np.array([[wxy[0], wxy[1], [0, 24, 0]] for wxy in wells_xy])

    np.save(jp(mod_dir, 'grid', 'iw'), wells_data)  # Saves injection wells stress period data
    np.save(jp(mod_dir, 'grid', 'pw'), pumping_well)  #


if __name__ == '__name__':
    well_maker()

