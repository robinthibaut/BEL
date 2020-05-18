#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import flopy
import meshio
import numpy as np
from flopy.export import vtk

import bel.toolbox.file_ops as fops
import bel.toolbox.mesh_ops as mops

rn = 'test'
results_dir = jp(os.getcwd(), 'bel', 'hydro', 'results', rn)
vtk_dir = jp(results_dir, 'vtk')
fops.dirmaker(vtk_dir)

# %% Load flow model
m_load = jp(results_dir, 'whpa.nam')
flow_model = fops.load_flow_model(m_load, model_ws=results_dir)
delr = flow_model.modelgrid.delr  # thicknesses along rows
delc = flow_model.modelgrid.delc  # thicknesses along column
xyz_vertices = flow_model.modelgrid.xyzvertices
# I'll be working with hexahedron, vtk type = 12
blocks2d = mops.blocks_from_rc(delc, delr)
blocks = mops.blocks_from_rc_3d(delc, delr)
blocks3d = blocks.reshape(-1, 3)


def flow_vtk():
    """
    Export flow model packages and computed heads to vtk format

    """
    dir_fv = jp(results_dir, 'vtk', 'flow')
    fops.dirmaker(dir_fv)
    flow_model.export(dir_fv, fmt='vtk')
    vtk.export_heads(flow_model,
                     jp(results_dir, 'whpa.hds'),
                     jp(results_dir, 'vtk', 'flow'),
                     binary=True, kstpkper=(0, 0))


flow_vtk()

# %% Load transport model
mt_load = jp(results_dir, 'whpa.mtnam')
transport_model = fops.load_transport_model(mt_load, flow_model, model_ws=results_dir)
ucn_files = [jp(results_dir, 'MT3D00{}.UCN'.format(i)) for i in range(1, 7)]  # Files containing concentration
ucn_obj = [flopy.utils.UcnFile(uf) for uf in ucn_files]  # Load them
times = [uo.get_times() for uo in ucn_obj]  # Get time steps
concs = np.array([uo.get_alldata() for uo in ucn_obj])  # Get all data


def transport_vtk():
    """
    Export transport package attributes to vtk format

    """
    dir_tv = jp(results_dir, 'vtk', 'transport')
    fops.dirmaker(dir_tv)
    transport_model.export(dir_tv, fmt='vtk')


transport_vtk()


# %% Export UCN to vtk

def stacked_conc_vtk():
    """Stack component concentrations for each time step and save vtk"""
    conc_dir = jp(results_dir, 'vtk', 'transport', 'concentration')
    fops.dirmaker(conc_dir)
    # First replace 1e+30 value (inactive cells) by 0.
    conc0 = np.abs(np.where(concs == 1e30, 0, concs))
    # Cells configuration for 3D blocks
    cells = [("quad", np.array([list(np.arange(i * 4, i * 4 + 4))])) for i in range(len(blocks))]
    for j in range(concs.shape[1]):
        # Stack the concentration of each component at each time step to visualize them in one plot.
        array = np.zeros(blocks.shape[0])
        for i in range(1, 7):
            # fliplr is necessary as the reshape modifies the array structure
            cflip = np.fliplr(conc0[i - 1, j]).reshape(-1)
            array += cflip  # Stack all components
        dic_conc = {'conc': array}
        # Use meshio to export the mesh
        meshio.write_points_cells(
            jp(conc_dir, 'cstack_{}.vtk'.format(j)),
            blocks3d,
            cells,
            # Optionally provide extra data on points, cells, etc.
            # point_data=point_data,
            cell_data=dic_conc,
            # field_data=field_data
        )


stacked_conc_vtk()

# %% Plot modpath


def particles_vtk():
    back_dir = jp(results_dir, 'vtk', 'backtrack')
    fops.dirmaker(back_dir)

    # mp_reloaded = backtrack(flowmodel=flow_model, exe_name='', load=True)
    # load the endpoint data
    endfile = jp(results_dir, 'whpa_mp.mpend')
    endobj = flopy.utils.EndpointFile(endfile)
    ept = endobj.get_alldata()

    # load the pathline data
    pthfile = jp(results_dir, 'whpa_mp.mppth')
    pthobj = flopy.utils.PathlineFile(pthfile)
    plines = pthobj.get_alldata()

    # load the time series
    tsfile = jp(results_dir, 'whpa_mp.timeseries')
    tso = flopy.utils.modpathfile.TimeseriesFile(tsfile)
    ts = tso.get_alldata()

    n_particles = len(ts)
    n_t_stp = ts[0].shape[0]
    time_steps = ts[0].time

    points_x = np.array([ts[i].x for i in range(len(ts))])
    points_y = np.array([ts[i].y for i in range(len(ts))])
    points_z = np.array([ts[i].z for i in range(len(ts))])

    for i in range(n_t_stp):
        xs = points_x[:, i]
        ys = points_y[:, i]
        zs = points_z[:, i]*0  # Replace elevation by 0 to project them in the surface
        xyz_particles_t_i = \
            np.vstack((xs, ys, zs)).T.reshape(-1, 3)
        cell_point = [("vertex", np.array([[i]])) for i in range(n_particles)]
        meshio.write_points_cells(
            jp(back_dir, 'particles_t{}.vtk'.format(i)),
            xyz_particles_t_i,
            cells=cell_point,
            # Optionally provide extra data on points, cells, etc.
            # point_data=point_data,
            point_data={'time': np.ones(n_particles)*time_steps[i]},
            # field_data=field_data
        )


particles_vtk()

# %% Export wells objects as vtk


def wels_vtk():
    pw = np.load(jp(os.getcwd(), 'bel', 'hydro', 'grid', 'pw.npy'), allow_pickle=True)[0, :2].tolist()
    iw = np.load(jp(os.getcwd(), 'bel', 'hydro', 'grid', 'iw.npy'), allow_pickle=True)[:, :2].tolist()

    wels = np.concatenate((iw, [pw]), axis=0)

    cell_point = [("vertex", np.array([[i]])) for i in range(len(wels))]
    point_data = np.array([np.linalg.norm(w - pw) for w in wels])

    meshio.write_points_cells(
        jp(vtk_dir, 'wels.vtk'),
        wels,
        cells=cell_point,
        point_data={'d_pw': point_data},
    )


wels_vtk()