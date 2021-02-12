#  Copyright (c) 2021. Robin Thibaut, Ghent University
"""
Grid geometry parameters need to be passed around modules.
Defining data classes allows to avoid declaring those parameters several times.
"""

import os
import platform
from dataclasses import dataclass
from os.path import dirname, join

import numpy as np


class machine(object):
    computer: str = platform.node()


class setup:
    @dataclass
    class directories:
        """Define main directories and file names"""

        # Content directory
        main_dir: str = dirname(os.path.abspath(__file__))
        hydro_res_dir: str = join(main_dir, 'storage', 'forwards')
        forecasts_dir: str = join(main_dir, 'storage', 'forecasts')
        grid_dir: str = join(main_dir, 'spatial', 'parameters')

    @dataclass
    class files:
        """Class to keep track of important file names"""
        # Output file names
        project_name: str = 'whpa'

        hk_file: str = 'hk0.npy'
        predictor_file: str = 'bkt.npy'
        target_file: str = 'pz.npy'

        output_files = [hk_file, predictor_file, target_file]

        sgems_file: str = 'hd.sgems'
        command_file: str = 'sgsim_commands.py'

        sgems_family = [sgems_file, command_file, hk_file]

    @dataclass
    class grid_dimensions:
        """Class for keeping track of grid dimensions"""
        x_lim: float = 1500.
        y_lim: float = 1000.
        z_lim: float = 1.

        dx: float = 10.  # Block x-dimension
        dy: float = 10.  # Block y-dimension
        dz: float = 10.  # Block z-dimension

        xo: float = 0.
        yo: float = 0.
        zo: float = 0.

        nrow: int = y_lim // dy  # Number of rows
        ncol: int = x_lim // dx  # Number of columns
        nlay: int = 1  # Number of layers

        # Refinement parameters around the pumping well.
        r_params = np.array(
            [[9, 150],  # 150 meters from the pumping well coordinates, grid cells will have dimensions 9x9
             [8, 100],
             [7, 90],
             [6, 80],
             [5, 70],
             [4, 60],
             [3, 50],
             [2.5, 40],
             [2, 30],
             [1.5, 20],
             [1, 10]])  # 10 meters from the pumping well coordinates, grid cells will have dimensions 1*1

    @dataclass
    class focus:
        """Geometry of the focused area on the main grid, enclosing all wells, as to reduce computation time"""
        x_range = [800, 1150]
        y_range = [300, 700]
        cell_dim: float = 4  # Defines cell dimensions for the signed distance computation.

    @dataclass
    # self.cols = ['w', 'g', 'r', 'c', 'm', 'y']
    class wells:
        """Wells coordinates"""
        wells_data = {
            'pumping0':
                {'coordinates': [1000, 500],
                 'rates': [-1000, -1000, -1000],
                 'color': 'w'},
            'injection0':
                {'coordinates': [950, 450],
                 'rates': [0, 24, 0],
                 'color': 'b'},
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

        combination = np.arange(1, len(wells_data))  # Injection wells in use for prediction (default: all)
        colors = ['w', 'g', 'r', 'c', 'm', 'y']

    @dataclass
    class forecast:
        n_posts: int = 200

    @dataclass
    class predictor:
        n_pc: int = 50
        n_tstp: int = 200

    @dataclass
    class target:
        n_pc: int = 30
