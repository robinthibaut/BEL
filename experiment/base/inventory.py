#  Copyright (c) 2020. Robin Thibaut, Ghent University
"""
Grid geometry parameters need to be passed around modules.
Defining data classes allows to avoid declaring those parameters several times.
"""

import os
from dataclasses import dataclass
from os.path import dirname, join

import numpy as np


@dataclass
class Directories:
    main_dir = dirname(dirname(os.path.abspath(__file__)))
    grid_dir = join(main_dir, 'grid', 'parameters')

@dataclass
class GridDimensions:
    """Class for keeping track of grid dimensions"""
    x_lim: int = 1500
    y_lim: int = 1000

    dx: int = 10  # Block x-dimension
    dy: int = 10  # Block y-dimension
    dz: int = 10  # Block z-dimension

    nrow: int = y_lim // dy  # Number of rows
    ncol: int = x_lim // dx  # Number of columns
    nlay: int = 1  # Number of layers

    # Refinement parameters around the pumping wel.
    r_params = np.array([[9, 150],  # 150 meters from the pumping wel coordinates, grid cells will have dimensions 9x9
                         [8, 100],
                         [7, 90],
                         [6, 80],
                         [5, 70],
                         [4, 60],
                         [3, 50],
                         [2.5, 40],
                         [2, 30],
                         [1.5, 20],
                         [1, 10]])  # 10 meters from the pumping wel coordinates, grid cells will have dimensions 1*1


@dataclass
class Wels:
    """Wels coordinates"""
    wels_data = {
        'pumping0':
            {'coordinates': [1000, 500],
             'rates': [-1000, -1000, -1000]},
        'injection0':
            {'coordinates': [950, 450],
             'rates': [0, 24, 0]},
        'injection1':
            {'coordinates': [930, 560],
             'rates': [0, 24, 0]},
        'injection2':
            {'coordinates': [900, 505],
             'rates': [0, 24, 0]},
        'injection3':
            {'coordinates': [1068, 515],
             'rates': [0, 24, 0]},
        'injection4':
            {'coordinates': [1030, 580],
             'rates': [0, 24, 0]},
        'injection5':
            {'coordinates': [1050, 470],
             'rates': [0, 24, 0]}
    }


@dataclass
class Focus:
    """Geometry of the focused area on the main grid, englobing all wells, as to reduce computation time"""
    x_range = [800, 1150]
    y_range = [300, 700]
    cell_dim = 1


