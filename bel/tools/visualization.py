import os
from os.path import join as jp

import matplotlib.pyplot as plt
import flopy

from bel.tools.toolbox import datread, load_flow_model, load_transport_model

rn = '2b3e08a97e774b62889d12c6358999ba'
results_dir = jp(os.getcwd(), 'results', rn)

m_load = jp(results_dir, 'whpa.nam')
flow_model = load_flow_model(m_load, model_ws=results_dir)
mt_load = jp(results_dir, 'whpa.mtnam')
transport_model = load_transport_model(mt_load, flow_model, model_ws=results_dir)

# let's take a look at our grid
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
mapview = flopy.plot.PlotMapView(model=flow_model, extent=(900, 1100, 400, 600))
ibound = mapview.plot_ibound()
# linecollection = mapview.plot_grid()
wel = mapview.plot_bc("WEL")
plt.show()

# plot head
fname = jp(results_dir, 'whpa.hds')
hdobj = flopy.utils.HeadFile(fname)
head = hdobj.get_data()
# levels = np.arange(10, 30, .5)

fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(1, 2, 1, aspect='equal')
ax.set_title('plot_array()')
mapview = flopy.plot.PlotMapView(model=flow_model)
quadmesh = mapview.plot_ibound()
quadmesh = mapview.plot_array(head, masked_values=[999.], alpha=0.5)
mapview.plot_bc("WEL")
plt.show()

ax = fig.add_subplot(1, 2, 2, aspect='equal')
ax.set_title('contour_array()')
mapview = flopy.plot.PlotMapView(model=flow_model)
quadmesh = mapview.plot_ibound()
mapview.plot_bc("WEL")
contour_set = mapview.contour_array(head, masked_values=[999.], levels=20)
plt.show()

# Plot modpath

# load the endpoint data
endfile = jp(results_dir, 'whpa_mp.mpend')
endobj = flopy.utils.EndpointFile(endfile)
ept = endobj.get_alldata()

# load the pathline data
pthfile = jp(results_dir, 'whpa_mp.mppth')
pthobj = flopy.utils.PathlineFile(pthfile)
plines = pthobj.get_alldata()

# plot the data
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(1, 1, 1, aspect='equal')
ax.set_title('plot_array()')
mapview = flopy.plot.PlotMapView(model=flow_model)
quadmesh = mapview.plot_ibound()
quadmesh = mapview.plot_array(head, masked_values=[999.], alpha=0.5)
mapview.plot_endpoint(ept)
plt.show()

obs = datread(jp(results_dir, 'MT3D001.OBS'), header=2)[:, 1:]
plt.plot(obs[:,0], obs[:,1])
plt.show()