#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import numpy as np
from pysgems.algo.sgalgo import XML
from pysgems.dis.sgdis import Discretize
from pysgems.io.sgio import PointSet
from pysgems.plot.sgplots import Plots
from pysgems.sgems import sg

from bel.toolbox.file_ops import datread


def transform(f):
    """
    Transforms the values of the statistical_simulation simulations into meaningful data
    """

    k_mean = np.random.uniform(1.4, 2)  # Hydraulic conductivity mean between x and y in m/d.
    k_std = 0.4

    ff = f * k_std + k_mean

    return 10 ** ff


def sgsim(model_ws, grid_dir):
    # %% Initiate sgems pjt
    pjt = sg.Sgems(project_name='sgsim', project_wd=grid_dir, res_dir=model_ws)

    # %% Load hard data point set

    data_dir = grid_dir
    dataset = 'wels.eas'
    file_path = jp(data_dir, dataset)

    hd = PointSet(project=pjt, pointset_path=file_path)

    hku = 1 + np.random.rand(len(hd.dataframe))
    hd.dataframe['hd'] = hku

    hd.export_01('hd')  # Exports in binary

    # %% Generate grid. Grid dimensions can automatically be generated based on the data points
    # unless specified otherwise, but cell dimensions dx, dy, (dz) must be specified
    Discretize(project=pjt, dx=10, dy=10, xo=0, yo=0, x_lim=1500, y_lim=1000)

    # %% Display point coordinates and grid
    pl = Plots(project=pjt)
    pl.plot_coordinates()

    # %% Which feature are available
    print(pjt.point_set.columns)

    # %% Load your algorithm xml file in the 'algorithms' folder.
    dir_path = os.path.abspath(__file__ + "/..")
    algo_dir = jp(dir_path, 'algorithms')
    al = XML(project=pjt, algo_dir=algo_dir)
    al.xml_reader('bel_sgsim')

    # %% Modify xml below:
    al.xml_update('Seed', 'value', str(np.random.randint(1e9)))

    # %% Write python script
    pjt.write_command()

    # %% Run sgems
    pjt.run()
    # Plot 2D results
    pl.plot_2d(save=True)

    opl = jp(model_ws, 'results.grid')  # Output file location.

    matrix = datread(opl, start=3)  # Grid information directly derived from the output file.
    matrix = np.where(matrix == -9966699, np.nan, matrix)

    tf = np.vectorize(transform)
    matrix = tf(matrix)

    matrix = matrix.reshape((pjt.dis.nrow, pjt.dis.ncol))

    matrix = np.flipud(matrix)

    np.save(jp(model_ws, 'hk0'), matrix)  # Save the un-discretized hk grid

    return matrix, centers
