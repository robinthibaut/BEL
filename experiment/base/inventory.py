#  Copyright (c) 2020. Robin Thibaut, Ghent University
"""
Grid geometry parameters need to be passed around modules.
Defining data classes allows to avoid declaring those parameters several times.
"""

import os
import platform
from dataclasses import dataclass
from os.path import dirname, join

import numpy as np


class Machine(object):
    computer = platform.node()


class MySetup:

    @dataclass
    class Directories:
        """Define main directories"""
        main_dir = dirname(dirname(os.path.abspath(__file__)))

        print(f'Working on {Machine.computer}')
        hydro_res_dir = join(main_dir, 'storage', 'forwards')

        # In future version I'll put the data under VCS and push to GitHub
        # if Machine.computer == 'MacBook-Pro.local':
        #     hydro_res_dir = '/Users/robin/OneDrive - UGent/Project-we13c420/experiment/hydro/results'
        # if Machine.computer == 'Yippee-Ki-yay-PC':
        #     hydro_res_dir = 'C:/Users/robin/OneDrive - UGent/Project-we13c420/experiment/hydro/results'
        # else:
        #     hydro_res_dir = join(main_dir, 'hydro', 'results')

        forecasts_dir = join(main_dir, 'storage', 'forecasts')
        grid_dir = join(main_dir, 'grid', 'parameters')

    @dataclass
    class FileNames:
        """Class to keep track of important file names"""
        pass

    @dataclass
    class GridDimensions:
        """Class for keeping track of grid dimensions"""
        x_lim: float = 1500
        y_lim: float = 1000
        z_lim: float = 1

        dx: float = 10  # Block x-dimension
        dy: float = 10  # Block y-dimension
        dz: float = 10  # Block z-dimension

        xo: float = 0
        yo: float = 0
        zo: float = 0

        nrow: int = y_lim // dy  # Number of rows
        ncol: int = x_lim // dx  # Number of columns
        nlay: int = 1  # Number of layers

        # Refinement parameters around the pumping wel.
        r_params = np.array(
            [[9, 150],  # 150 meters from the pumping wel coordinates, grid cells will have dimensions 9x9
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
    # self.cols = ['w', 'g', 'r', 'c', 'm', 'y']
    class Wels:
        """Wels coordinates"""
        wels_data = {
            'pumping0':
                {'coordinates': [1000, 500],
                 'rates': [-1000, -1000, -1000],
                 'color': 'k'},
            'injection0':
                {'coordinates': [950, 450],
                 'rates': [0, 24, 0],
                 'color': 'w'},
            'injection1':
                {'coordinates': [930, 560],
                 'rates': [0, 24, 0],
                 'color': 'g'},
            'injection2':
                {'coordinates': [900, 505],
                 'rates': [0, 24, 0],
                 'color': 'r'},
            'injection3':
                {'coordinates': [1068, 515],
                 'rates': [0, 24, 0],
                 'color': 'c'},
            'injection4':
                {'coordinates': [1030, 580],
                 'rates': [0, 24, 0],
                 'color': 'm'},
            'injection5':
                {'coordinates': [1050, 470],
                 'rates': [0, 24, 0],
                 'color': 'y'}
        }

        combination = np.arange(1, len(wels_data))  # Injection wells in use for prediction (default: all)

    @dataclass
    class Focus:
        """Geometry of the focused area on the main grid, enclosing all wells, as to reduce computation time"""
        x_range = [800, 1150]
        y_range = [300, 700]
        cell_dim = 4  # Defines cell dimensions for the signed distance computation.

    @dataclass
    class Forecast:
        n_posts = 500
