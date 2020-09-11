#  Copyright (c) 2020. Robin Thibaut, Ghent University

import numpy as np
from experiment.base.inventory import MySetup as base
from experiment.goggles.vtkio import ModelVTK
from experiment.goggles.visualization import Plot
from experiment.toolbox import filesio

if __name__ == '__main__':
    x_lim, y_lim, grf = base.Focus.x_range, base.Focus.y_range, base.Focus.cell_dim
    # Initiate Plot instance
    mp = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf, wel_comb=base.Wels.combination)

    # VTK
    mi = ModelVTK(base=base, folder='illustration')
    mi.flow_vtk()
    mi.transport_vtk()
    mi.conc_vtk()
    mi.particles_vtk(path=1)
    mi.wels_vtk()




