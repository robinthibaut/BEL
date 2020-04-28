#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import flopy
import matplotlib.pyplot as plt
import meshio
import numpy as np
from flopy.export import vtk

import bel.toolbox.file_ops as fops
import bel.toolbox.mesh_ops as mops
from bel.hydro.backtracking.modpath import backtrack

rn = 'example_illustration'
results_dir = jp(os.getcwd(), 'bel', 'hydro', 'results', rn)

# %% Load flow model
m_load = jp(results_dir, 'whpa.nam')
flow_model = fops.load_flow_model(m_load, model_ws=results_dir)
delr = flow_model.modelgrid.delr
delc = flow_model.modelgrid.delc
xyz_vertices = flow_model.modelgrid.xyzvertices
# I'll be working with hexahedron, vtk type = 12
blocks2d = mops.blocks_from_rc(delc, delr)
blocks = mops.blocks_from_rc_3d(delc, delr)
blocks3d = blocks.reshape(-1, 3)


def flow_vtk(export=False):
    flow_model.export(jp(results_dir, 'vtk', 'flow'), fmt='vtk')
    vtk.export_heads(flow_model,
                     jp(results_dir, 'whpa.hds'),
                     jp(results_dir, 'vtk', 'flow'),
                     binary=True, kstpkper=(0, 0))


# %% Load transport model
mt_load = jp(results_dir, 'whpa.mtnam')
transport_model = fops.load_transport_model(mt_load, flow_model, model_ws=results_dir)
ucn_files = [jp(results_dir, 'MT3D00{}.UCN'.format(i)) for i in range(1, 7)]
ucn_obj = [flopy.utils.UcnFile(uf) for uf in ucn_files]
times = [uo.get_times() for uo in ucn_obj]
concs = np.array([uo.get_alldata() for uo in ucn_obj])


def transport_vtk():
    transport_model.export(jp(results_dir, 'vtk', 'transport'), fmt='vtk')


# %% Plot modpath

mp_reloaded = backtrack(flowmodel=flow_model, exe_name='', load=True)
# load the endpoint data
endfile = jp(results_dir, 'whpa_mp.mpend')
endobj = flopy.utils.EndpointFile(endfile)
ept = endobj.get_alldata()
vtk.export_array(model=flow_model,
                 array=ept,
                 output_folder=jp(results_dir, 'vtk', 'backtrack'),
                 name='ept')

# load the pathline data
pthfile = jp(results_dir, 'whpa_mp.mppth')
pthobj = flopy.utils.PathlineFile(pthfile)
plines = pthobj.get_alldata()

# load the time series
tsfile = jp(results_dir, 'whpa_mp.timeseries')
tso = flopy.utils.modpathfile.TimeseriesFile(tsfile)
ts = tso.get_alldata()

# plot the data
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(1, 1, 1, aspect='equal')
ax.set_title('plot_array()')
mapview = flopy.plot.PlotMapView(model=flow_model)
mapview.plot_ibound()
# mapview.plot_array(head, masked_values=[999.], alpha=0.5)
mapview.plot_endpoint(ept)
plt.show()


# %% Export UCN to vtk
cells = [("quad", np.array([list(np.arange(i*4, i*4+4))])) for i in range(len(blocks))]
for i in range(1, 7):
    conc = np.array([np.fliplr(concs[i-1, j]).reshape(-1) for j in range(len(concs[i-1]))])
    conc0 = np.abs(np.where(conc == 1e30, 0, conc))
    dic_conc = [{'conc{}'.format(i): conc0[i-1]} for i in range(len(conc0))]
    array = {}
    [array.update(c) for c in dic_conc]
    meshio.write_points_cells(
        jp(results_dir, 'vtk', 'transport', 'MT3D00{}.vtk'.format(i)),
        blocks3d,
        cells,
        # Optionally provide extra data on points, cells, etc.
        # point_data=point_data,
        cell_data=array,
        # field_data=field_data
        )

# %%
conc0 = np.abs(np.where(concs == 1e30, 0, concs))
array = {}
for i in range(1, 7):
    cflip = np.fliplr(conc0[i-1, 2]).reshape(-1)
    dic_conc = {'conc{}'.format(i): cflip}
    array.update(dic_conc)

meshio.write_points_cells(
    jp(results_dir, 'vtk', 'transport', 'concentrations.vtk'),
    blocks3d,
    cells,
    # Optionally provide extra data on points, cells, etc.
    # point_data=point_data,
    cell_data=array,
    # field_data=field_data
    )

# %%
conc0 = np.abs(np.where(concs == 1e30, 0, concs))
array = np.zeros(blocks.shape[0])
for i in range(1, 7):
    cflip = np.fliplr(conc0[i-1, 2]).reshape(-1)
    array += cflip

dic_conc = {'conc': array}

meshio.write_points_cells(
    jp(results_dir, 'vtk', 'transport', 'concentrations_stacked.vtk'),
    blocks3d,
    cells,
    # Optionally provide extra data on points, cells, etc.
    # point_data=point_data,
    cell_data=dic_conc,
    # field_data=field_data
    )
