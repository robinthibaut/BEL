#  Copyright (c) 2021. Robin Thibaut, Ghent University

from bel4ed.config import Setup as base
from bel4ed.goggles import ModelVTK

if __name__ == "__main__":
    x_lim, y_lim, grf = base.Focus.x_range, base.Focus.y_range, base.Focus.cell_dim
    # Initiate Plot instance
    # VTK
    mi = ModelVTK(base=base, folder="818bf1676c424f76b83bd777ae588a1d")
    mi.flow_vtk()
    mi.transport_vtk()
    mi.conc_vtk()
    mi.particles_vtk(path=1)
    mi.wells_vtk()
