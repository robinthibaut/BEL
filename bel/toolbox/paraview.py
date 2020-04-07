import os
from os.path import join as jp
import numpy as np

import meshio

from bel.toolbox.tools import FileOps
from bel.toolbox.tools import MeshOps

ops = FileOps()
mo = MeshOps()

rn = 'illustration'
results_dir = jp(os.getcwd(), 'bel', 'hydro', 'results', rn)
m_load = jp(results_dir, 'whpa.nam')
flow_model = ops.load_flow_model(m_load, model_ws=results_dir)

delr = flow_model.modelgrid.delr
delc = flow_model.modelgrid.delc