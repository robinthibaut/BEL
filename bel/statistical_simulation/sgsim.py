#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import numpy as np
from scipy.spatial import distance_matrix

import bel.statistical_simulation.sgems as sg
from bel.toolbox.mesh_ops import blocks_from_rc


def sgsim(model_ws):
    mod_dir = os.getcwd()  # Module directory
    main_dir = os.path.dirname(mod_dir)
    grid_dir = jp(main_dir, 'hydro', 'grid')

    x_lim = 1500
    y_lim = 1000
    dx = 10  # Block x-dimension
    dy = 10  # Block y-dimension
    dz = 10  # Block z-dimension
    xo, yo, zo = 0, 0, 0
    nrow = y_lim // dy  # Number of rows
    ncol = x_lim // dx  # Number of columns
    nlay = 1
    along_r = np.ones(ncol) * dx  # Size of each cell along y-dimension - rows
    along_c = np.ones(nrow) * dy  # Size of each cell along x-dimension - columns

    blocks = blocks_from_rc(along_c, along_r)
    centers = np.array([np.mean(b, axis=0) for b in blocks])

    def my_rc(xy):
        """
        :param xy: [x, y]
        :return:
        """
        rn = np.array([xy])
        dm = np.flipud(distance_matrix(rn, centers).reshape(nrow, ncol))
        rc = np.where(dm == np.amin(dm))
        return rc[0][0], rc[1][0]

    def my_node(xy):
        """
        :param xy: [x, y]
        :return:
        """
        rn = np.array([xy])
        dm = np.flipud(distance_matrix(rn, centers).reshape(nrow, ncol)).flatten()
        cell = np.where(dm == np.amin(dm))
        return cell[0][0]

    pwa = np.array(np.load(jp(grid_dir, 'pw.npy'), allow_pickle=True))[:, :2]  # Pumping well location
    iwa = np.array(np.load(jp(grid_dir, 'iw.npy'), allow_pickle=True))[:, :2]  # Injection wells locations
    wells_loc = np.concatenate((pwa, iwa), axis=0)

    op = 'hk'  # Simulations output name

    if os.path.exists(jp(model_ws, '{}0.npy').format(op)):  # If re-using an older model
        val = np.load(jp(model_ws, '{}0.npy').format(op))
        return val
    else:
        sgems = sg.SGEMS()
        nr = 1  # Number of realizations.
        # Extracts wells nodes number in statistical_simulation grid, to fix their simulated value.
        wells_nodes_sgems = [my_node(w) for w in wells_loc]
        # Hard data node
        # fixed_nodes = [[pwnode_sg, 2], [iw1node_sg, 1], [iw2node_sg, 1.5], [iw3node_sg, 0.7], [iw4node_sg, 1.2]]
        # I now arbitrarily attribute a random value between 1 and 2 to the well nodes
        fixed_nodes = [[w, 1 + np.random.rand()] for w in wells_nodes_sgems]

        sgrid = [ncol,
                 nrow,
                 nlay,
                 dx,
                 dy,
                 dz,
                 xo, yo, zo]  # Grid information

        seg = [50, 50, 50, 0, 0, 0]  # Search ellipsoid geometry

        sgems.gaussian_simulation(op_folder=model_ws.replace('\\', '//'),
                                  simulation_name='hk',
                                  output=op,
                                  grid=sgrid,
                                  fixed_nodes=fixed_nodes,
                                  algo='sgsim',
                                  number_realizations=nr,
                                  seed=np.random.randint(1000000),
                                  kriging_type='Simple Kriging (SK)',
                                  trend=[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  local_mean=0,
                                  hard_data_grid='hd',
                                  hard_data_property='hard',
                                  assign_hard_data=1,
                                  max_conditioning_data=15,
                                  search_ellipsoid_geometry=seg,
                                  target_histogram_flag=0,
                                  target_histogram=[0, 0, 0, 0],
                                  variogram_nugget=0,
                                  variogram_number_structures=1,
                                  variogram_structures_contribution=[1],
                                  variogram_type=['Spherical'],
                                  range_max=[100],
                                  range_med=[50],
                                  range_min=[25],
                                  angle_x=[0],
                                  angle_y=[0],
                                  angle_z=[0])

        opl = jp(model_ws, '{}.grid'.format(op))  # Output file location.

        hk = sg.so(opl)  # Grid information directly derived from the output file.

        k_mean = np.random.uniform(1.4, 2)  # Hydraulic conductivity mean between x and y in m/d.

        k_std = 0.4

        hkp = np.copy(hk)

        hk_array = [sg.transform(h, k_mean, k_std) for h in hkp]

        # Setting the hydraulic conductivity matrix.
        hk_array = [np.reshape(h, (nlay, nrow, ncol)) for h in hk_array]

        # Flip to correspond to statistical_simulation grid ! hk_array is now a list of the
        # arrays of conductivities for each realization.
        hk_array = [np.fliplr(hk_array[h]) for h in range(0, nr, 1)]

        np.save(jp(model_ws, op + '0'), hk_array[0])  # Save the un-discretized hk grid

        # Flattening hk_array to plot it
        fl = [item for sublist in hk_array[0] for item in sublist]
        fl2 = [item for sublist in fl for item in sublist]
        val = []
        for n in range(nlay):  # Adding 'nlay' times so all layers get the same conductivity.
            val.append(fl2)
        val = [item for sublist in val for item in sublist]  # Flattening

        return val, centers

