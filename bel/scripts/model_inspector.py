#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import flopy
import matplotlib.pyplot as plt
from flopy.export import vtk

import bel.toolbox.file_ops as fops

rn = 'example_illustration'
results_dir = jp(os.getcwd(), 'bel', 'hydro', 'results', rn)

# Load flow model
m_load = jp(results_dir, 'whpa.nam')
flow_model = fops.load_flow_model(m_load, model_ws=results_dir)
flow_model.export(jp(results_dir, 'vtk', 'flow'), fmt='vtk')
vtk.export_heads(flow_model, jp(results_dir, 'whpa.hds'), jp(results_dir, 'vtk', 'flow'), binary=True, kstpkper=(0, 0))
# Wells are ordered by node number
spd = flow_model.wel.stress_period_data.df

# Load transport model
mt_load = jp(results_dir, 'whpa.mtnam')
transport_model = fops.load_transport_model(mt_load, flow_model, model_ws=results_dir)
transport_model.export(jp(results_dir, 'vtk', 'transport'), fmt='vtk')
# ucn_files = [jp(results_dir, 'MT3D00{}.UCN'.format(i)) for i in range(1, 7)]
# ucn_obj = [flopy.utils.UcnFile(uf) for uf in ucn_files]
# times = [uo.get_times() for uo in ucn_obj]
# concs = [uo.get_alldata() for uo in ucn_obj]

# # let's take a look at our grid
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# mapview = flopy.plot.PlotMapView(model=flow_model, extent=(900, 1100, 400, 600))
# ibound = mapview.plot_ibound()
# linecollection = mapview.plot_grid()
# wel = mapview.plot_bc("WEL")
# plt.show()
#
# # let's take a look at our grid
# fig_mt = plt.figure(figsize=(8, 8))
# ax_mt = fig_mt.add_subplot(1, 1, 1, aspect='equal')
# mapview_mt = flopy.plot.PlotMapView(model=transport_model, extent=(0, 1500, 0, 1000))
# ibound_mt = mapview_mt.plot_ibound()
# plt.show()

# plot head
# fname = jp(results_dir, 'whpa.hds')
# hdobj = flopy.utils.HeadFile(fname)
# head = hdobj.get_data()
# # levels = np.arange(10, 30, .5)

# fig = plt.figure(figsize=(15, 10))
#
# ax = fig.add_subplot(1, 2, 1, aspect='equal')
# ax.set_title('plot_array()')
# mapview = flopy.plot.PlotMapView(model=flow_model)
# mapview.plot_ibound()
# mapview.plot_array(head, masked_values=[999.], alpha=0.5)
# mapview.plot_bc("WEL")
# plt.show()
#
# ax = fig.add_subplot(1, 2, 2, aspect='equal')
# ax.set_title('contour_array()')
# mapview = flopy.plot.PlotMapView(model=flow_model)
# mapview.plot_ibound()
# mapview.plot_bc("WEL")
# contour_set = mapview.contour_array(head, masked_values=[999.], levels=20)
# plt.show()

# Plot modpath

mpnam = jp(results_dir, 'whpa_mp.mpnam')
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

obs = fops.datread(jp(results_dir, 'MT3D001.OBS'), header=2)[:, 1:]
plt.plot(obs[:, 0], obs[:, 1])
plt.show()
