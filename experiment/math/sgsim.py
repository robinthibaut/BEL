#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import numpy as np
from pysgems.algo.sgalgo import XML
from pysgems.dis.sgdis import Discretize
from pysgems.io.sgio import PointSet
from pysgems.sgems import sg

from experiment.base.inventory import MySetup
from experiment.toolbox.filesio import datread


def transform(f, k_mean, k_std):
    """
    Transforms the values of the statistical_simulation simulations into meaningful data
    :param: f: float: Simulation output
    :param: k_mean: float: Desired mean for the Hk field
    """

    ff = f * k_std + k_mean

    return 10 ** ff


def sgsim(model_ws, grid_dir, wells_hk=None):
    # %% Initiate sgems pjt
    pjt = sg.Sgems(project_name='sgsim', project_wd=grid_dir, res_dir=model_ws)

    # %% Load hard data point set

    data_dir = grid_dir
    dataset = 'wels.eas'
    file_path = jp(data_dir, dataset)

    hd = PointSet(project=pjt, pointset_path=file_path)

    if wells_hk is None:
        hku = 2. + np.random.rand(len(hd.dataframe))  # Fix hard data values at wells location
    else:
        hku = wells_hk

    if not os.path.exists(jp(model_ws, MySetup.Directories.sgems_file)):
        hd.dataframe['hd'] = hku
        hd.export_01('hd')  # Exports modified dataset in binary

    # %% Generate grid. Grid dimensions can automatically be generated based on the data points
    # unless specified otherwise, but cell dimensions dx, dy, (dz) must be specified
    gd = MySetup.GridDimensions()
    Discretize(project=pjt, dx=gd.dx, dy=gd.dy, xo=gd.xo, yo=gd.yo, x_lim=gd.x_lim, y_lim=gd.y_lim)

    # Get sgems grid centers coordinates:
    x = np.cumsum(pjt.dis.along_r) - pjt.dis.dx / 2
    y = np.cumsum(pjt.dis.along_c) - pjt.dis.dy / 2
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
    centers = np.stack((xv, yv), axis=2).reshape((-1, 2))

    if os.path.exists(jp(model_ws, 'hk0.npy')):
        hk0 = np.load(jp(model_ws, 'hk0.npy'))
        return hk0, centers

    # %% Display point coordinates and grid
    # pl = Plots(project=pjt)
    # pl.plot_coordinates()

    # %% Load your algorithm xml file in the 'algorithms' folder.
    dir_path = os.path.abspath(__file__ + "/..")
    algo_dir = jp(dir_path, 'algorithms')
    al = XML(project=pjt, algo_dir=algo_dir)
    al.xml_reader('bel_sgsim')

    # %% Modify xml below:
    al.xml_update('Seed', 'value', str(np.random.randint(1e9)), show=0)

    # %% Write python script
    pjt.write_command()

    # %% Run sgems
    pjt.run()
    # Plot 2D results
    # pl.plot_2d(save=True)

    opl = jp(model_ws, 'results.grid')  # Output file location.

    matrix = datread(opl, start=3)  # Grid information directly derived from the output file.
    matrix = np.where(matrix == -9966699, np.nan, matrix)

    k_mean = np.random.uniform(1.4, 2)  # Hydraulic conductivity mean between x and y in m/d.
    print(f'hk mean={10 ** k_mean}')
    k_std = 0.4  # Log value of the standard deviation

    tf = np.vectorize(transform)  # Transform values from log10
    matrix = tf(matrix, k_mean, k_std)  # Apply function to results

    matrix = matrix.reshape((pjt.dis.nrow, pjt.dis.ncol))  # reshape - assumes 2D !
    matrix = np.flipud(matrix)  # Flip to correspond to sgems

    # import matplotlib.pyplot as plt
    # extent = (pjt.dis.xo, pjt.dis.x_lim, pjt.dis.yo, pjt.dis.y_lim)
    # plt.imshow(np.log10(matrix), cmap='coolwarm', extent=extent)
    # plt.plot(pjt.point_set.raw_data[:, 0], pjt.point_set.raw_data[:, 1], 'k+', markersize=1, alpha=.7)
    # plt.colorbar()
    # plt.show()

    np.save(jp(model_ws, 'hk0'), matrix)  # Save the un-discretized hk grid

    return matrix, centers
