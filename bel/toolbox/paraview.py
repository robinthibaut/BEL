import os
from os.path import join as jp
import numpy as np

import meshio
import matplotlib.pyplot as plt

import diavatly
from bel.toolbox.file_ops import FileOps
from bel.toolbox.mesh_ops import MeshOps
from bel.toolbox.plots import Plot


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

# Now let's try in 3D

btest = np.array([[ 0.        ,  0.        ,  0.        ],
                 [ 0.        , 10.01430035,  0.        ],
                 [10.00226021, 10.01430035,  0.        ],
                 [10.00226021,  0.        ,  0.        ],
                  [ 0.        ,  0.        ,  -10        ],
                 [ 0.        , 10.01430035,  -10        ],
                 [10.00226021, 10.01430035,  -10        ],
                 [10.00226021,  0.        ,  -10        ]])

cells = [("hexahedron", np.array([[4, 5, 6, 7, 0, 1, 2, 3]]))]
meshio.write_points_cells(
    "foo3d.vtk",
    btest,
    cells,
    # Optionally provide extra data on points, cells, etc.
    # point_data=point_data,
    cell_data={'hk': np.array([1])},
    # field_data=field_data
    )

if __name__ == '__main__':
    pass
