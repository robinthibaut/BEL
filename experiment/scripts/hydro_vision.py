#  Copyright (c) 2020. Robin Thibaut, Ghent University

import numpy as np
from experiment.base.inventory import MySetup as base
from experiment.goggles.vtkio import ModelVTK
from experiment.goggles.visualization import Plot
from experiment.toolbox import filesio

if __name__ == '__main__':
    x_lim, y_lim, grf = base.Focus.x_range, base.Focus.y_range, base.Focus.cell_dim
    # Initiate Plot instance
    mp = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf, well_comb=base.Wells.combination)

    # VTK
    mi = ModelVTK(base=base, folder='818bf1676c424f76b83bd777ae588a1d')
    mi.flow_vtk()
    # mi.transport_vtk()
    # mi.conc_vtk()
    # mi.particles_vtk(path=1)
    # mi.wells_vtk()




