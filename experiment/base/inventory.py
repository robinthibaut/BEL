#  Copyright (c) 2020. Robin Thibaut, Ghent University
"""
Grid geometry parameters need to be passed around modules.
Defining data classes allows to avoid declaring those parameters several times.
"""

from dataclasses import dataclass

import numpy as np


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
    wels_coordinates = {'pumping1': [1000, 500],
                        'injection1': [950, 450],
                        'injection2': [930, 560],
                        'injection3': [900, 505],
                        'injection4': [1068, 515],
                        'injection5': [1030, 580],
                        'injection6': [1050, 470]}


@dataclass
class Focus:
    """Geometry of the focused area on the main grid, englobing all wells, as to reduce computation time"""
    x_range = [800, 1150]
    y_range = [300, 700]
    cell_dim = 1
