#  Copyright (c) 2021. Robin Thibaut, Ghent University

from experiment._core import setup as base
from experiment._visualization import ModelVTK

if __name__ == '__main__':
    x_lim, y_lim, grf = base.focus.x_range, base.focus.y_range, base.focus.cell_dim
    # Initiate Plot instance
    # VTK
    mi = ModelVTK(base=base, folder='818bf1676c424f76b83bd777ae588a1d')
    mi.flow_vtk()
    # mi.transport_vtk()
    # mi.conc_vtk()
    # mi.particles_vtk(path=1)
    # mi.wells_vtk()




