from os.path import join as jp
import numpy as np

import flopy

from bel.hydro.whpa.travelling_particles import tsp
from bel.processing.signed_distance import SignedDistance, get_centroids
from bel.toolbox.mesh_ops import MeshOps


def active_zone(modflowmodel):

    version = 'mt3d-usgs'
    namefile_ext = 'mtnam'
    ftlfilename = 'mt3d_link.ftl'
    # Extract working directory and name from modflow object
    model_ws = modflowmodel.model_ws
    modelname = modflowmodel.name

    # Initiate Mt3dms object
    mt = flopy.mt3d.Mt3dms(modflowmodel=modflowmodel,
                           ftlfilename=ftlfilename,
                           modelname=modelname,
                           model_ws=model_ws,
                           version=version,
                           namefile_ext=namefile_ext)

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
    xy_nodes_2d = np.reshape(xy_true, (nlay * nrow * ncol, 2))  # Flattens xy to correspond to node numbering

    wells_data = np.load(jp(model_ws, 'spd.npy'))  # Loads well stress period data
    wells_number = wells_data.shape[0]  # Total number of wells
    pumping_well_data = wells_data[0]  # Pumping well in first
    pw_lrc = pumping_well_data[0][:3]  # PW layer row column
    pw_node = int(dis.get_node(pw_lrc)[0])  # PW node number
    xy_pumping_well = xy_nodes_2d[pw_node]  # Get PW x, y coordinates (meters) from well node number

    injection_well_data = wells_data[1:]
    iw_nodes = [int(dis.get_node(w[0][:3])[0]) for w in injection_well_data]
    xy_injection_wells = [xy_nodes_2d[iwn] for iwn in iw_nodes]

    sdm = SignedDistance()
    sdm.xys = xy_true
    sdm.nrow = nrow
    sdm.ncol = ncol
    diffs = xy_injection_wells - xy_pumping_well
    dists = np.array(list(map(np.linalg.norm, diffs)))
    meas = 1 / np.log(dists)
    meas -= min(meas)
    meas /= max(meas)
    meas += 1
    meas *= 2
    xyw_scaled = diffs * meas.reshape(-1, 1) + xy_pumping_well
    poly_deli = tsp(xyw_scaled)
    poly_xyw = xyw_scaled[poly_deli]
    icbund = sdm.matrix_poly_bin(poly_xyw, outside=0, inside=1).reshape(nlay, nrow, ncol)
    mt_icbund_file = jp('grid', 'mt3d_icbund.npy')
    np.save(mt_icbund_file, icbund)
    icbund = np.load(mt_icbund_file)

    val_icbund = [item for sublist in icbund[0] for item in sublist]  # Flattening
    grf_dummy = 10
    nrow_dummy = int(np.round(np.cumsum(dis.delc)[-1])/grf_dummy)
    ncol_dummy = int(np.round(np.cumsum(dis.delr)[-1])/grf_dummy)
    array_dummy = np.zeros((nrow_dummy, ncol_dummy))
    xy_dummy = get_centroids(array_dummy, grf=grf_dummy)
    mo = MeshOps()
    inds = mo.matrix_paste(xy_dummy, xy_true)
    val_dummy = [val_icbund[k] for k in inds]  # Contains k values for refined grid
    val_dummy_r = np.reshape(val_dummy, (nrow_dummy, ncol_dummy))  # Reshape in n layers x n cells in refined grid.
    # cdx = np.reshape(xy_true, (nlay, nrow, ncol, 2))  # Coordinates of centroids of refined grid
    from bel.toolbox.plots import Plot
    po = Plot(grf=grf_dummy)
    po.whp(bkg_field_array=np.flipud(val_dummy_r),
           show_wells=True,
           show=True)

    from diavatly import model_map
    import matplotlib.pyplot as plt
    from bel.toolbox.mesh_ops import MeshOps
    mo = MeshOps()
    grid1 = mo.blocks_from_rc(np.ones(nrow_dummy)*grf_dummy, np.ones(ncol_dummy)*grf_dummy)
    model_map(grid1, vals=val_dummy, log=0)
    plt.plot(xy_pumping_well[0], xy_pumping_well[1], 'ko', markersize=2000)
    plt.show()

