#  Copyright (c) 2021. Robin Thibaut, Ghent University

from os.path import join as jp

import flopy
import matplotlib.pyplot as plt
import numpy as np
from diavatly import model_map

import experiment.visualization as mplot
from experiment.spatial import binary_polygon, get_centroids, grid_parameters
from experiment.config import Setup
from experiment.processing.target_handle import travelling_particles


def active_zone(modflowmodel):
    version = "mt3d-usgs"
    namefile_ext = "mtnam"
    ftlfilename = "mt3d_link.ftl"
    # Extract working directory and name from modflow object
    model_ws = modflowmodel.model_ws
    modelname = modflowmodel.name

    # Initiate Mt3dms object
    mt = flopy.mt3d.Mt3dms(
        modflowmodel=modflowmodel,
        ftlfilename=ftlfilename,
        modelname=modelname,
        model_ws=model_ws,
        version=version,
        namefile_ext=namefile_ext,
    )

    # Extract discretization info from modflow object
    dis = modflowmodel.dis  # DIS package
    nlay = modflowmodel.nlay
    nrow = modflowmodel.nrow
    ncol = modflowmodel.ncol
    # yxz_grid is an array of the coordinates of each node:
    # [[coordinates Y], [coordinates X], [coordinates Z]]
    yxz_grid = dis.get_node_coordinates()
    xy_true = []  # Convert to 2D array
    for yc in yxz_grid[0]:
        for xc in yxz_grid[1]:
            xy_true.append([xc, yc])
    # Flattens xy to correspond to node numbering
    xy_nodes_2d = np.reshape(xy_true, (nlay * nrow * ncol, 2))

    # Loads well stress period data
    wells_data = np.load(jp(model_ws, "spd.npy"))
    pumping_well_data = wells_data[0]  # Pumping well in first
    pw_lrc = pumping_well_data[0][:3]  # PW layer row column
    pw_node = int(dis.get_node(pw_lrc)[0])  # PW node number
    # Get PW x, y coordinates (meters) from well node number
    xy_pumping_well = xy_nodes_2d[pw_node]

    injection_well_data = wells_data[1:]
    iw_nodes = [int(dis.get_node(w[0][:3])[0]) for w in injection_well_data]
    xy_injection_wells = [xy_nodes_2d[iwn] for iwn in iw_nodes]

    sdm = grid_parameters()
    sdm.xys = xy_true
    sdm.nrow = nrow
    sdm.ncol = ncol

    # Given the euclidean distances from each injection well to the pumping well,
    # arbitrary parameters are chosen to expand the polygon defined by those welles around
    # the pumping well. This expanded polygon will then act as a mask to define the
    # transport active zone.
    diffs = xy_injection_wells - xy_pumping_well
    dists = np.array(list(map(np.linalg.norm, diffs)))  # Eucliden distances
    meas = 1 / np.log(dists)
    meas -= min(meas)
    meas /= max(meas)
    meas += 1
    meas *= 2

    # Expand the polygon
    xyw_scaled = diffs * meas.reshape(-1, 1) + xy_pumping_well
    poly_deli = travelling_particles(xyw_scaled)  # Get polygon delineation
    poly_xyw = xyw_scaled[poly_deli]  # Obtain polygon vertices
    # Assign 0|1 value
    icbund = binary_polygon(
        sdm.xys, sdm.nrow, sdm.ncol, poly_xyw, outside=0, inside=1
    ).reshape(nlay, nrow, ncol)

    mt_icbund_file = jp(Setup.Directories.grid_dir, "mt3d_icbund.npy")
    np.save(mt_icbund_file, icbund)  # Save active zone

    # Check what we've done: plot the active zone.
    val_icbund = [item for sublist in icbund[0] for item in sublist]  # Flattening
    # Define a dummy grid with large cell dimensions to plot on it.
    grf_dummy = 10
    nrow_dummy = int(np.round(np.cumsum(dis.delc)[-1]) / grf_dummy)
    ncol_dummy = int(np.round(np.cumsum(dis.delr)[-1]) / grf_dummy)
    array_dummy = np.zeros((nrow_dummy, ncol_dummy))
    xy_dummy = get_centroids(array_dummy, grf=grf_dummy)

    inds = experiment.spatial.spatial.matrix_paste(xy_dummy, xy_true)
    val_dummy = [val_icbund[k] for k in inds]  # Contains k values for refined grid
    # Reshape in n layers x n cells in refined grid.
    val_dummy_r = np.reshape(val_dummy, (nrow_dummy, ncol_dummy))

    # We have to use flipud for the matrix to correspond.
    mplot.whpa_plot(
        grf=grf_dummy,
        bkg_field_array=np.flipud(val_dummy_r),
        show_wells=True,
        show=True,
    )

    grid1 = experiment.spatial.spatial.blocks_from_rc(
        np.ones(nrow_dummy) * grf_dummy, np.ones(ncol_dummy) * grf_dummy
    )
    model_map(grid1, vals=val_dummy, log=0)
    plt.show()
