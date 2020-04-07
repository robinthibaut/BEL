import os
from os.path import join as jp
import numpy as np

import meshio
import matplotlib.pyplot as plt

import diavatly
from bel.toolbox.tools import FileOps, MeshOps, Plot


ops = FileOps()
mo = MeshOps()
po = Plot()

rn = 'illustration'
results_dir = jp(os.getcwd(), 'bel', 'hydro', 'results', rn)
m_load = jp(results_dir, 'whpa.nam')
flow_model = ops.load_flow_model(m_load, model_ws=results_dir)

# TODO: write issue for inverted delr and delc
delc = flow_model.modelgrid.delr
delr = flow_model.modelgrid.delc
xyz_vertices = flow_model.modelgrid.xyzvertices

# I'll be working with hexahedron, vtk type = 12
blocks2d = mo.blocks_from_rc(delr, delc)
blocks = mo.blocks_from_rc_3d(delr, delc)
blocks3d = blocks.reshape(-1, 3)
# Let's first try 2D export !
# Load hk array
hk = np.load(jp(results_dir, 'hk.npy')).reshape(-1)
cells = [("quad", np.array([list(np.arange(i*4, i*4+4))])) for i in range(len(blocks))]
mesh = meshio.Mesh(blocks3d, cells)
meshio.write_points_cells(
    "foo.vtk",
    blocks3d,
    cells,
    # Optionally provide extra data on points, cells, etc.
    # point_data=point_data,
    cell_data={'hk': hk},
    # field_data=field_data
    )

diavatly.model_map(blocks2d, hk, log=1)
plt.show()

#  Now let's try in 3D