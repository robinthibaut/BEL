#  Copyright (c) 2020. Robin Thibaut, Ghent University

from experiment.goggles.vtkio import ModelVTK
from experiment.goggles.visualization import Plot
from experiment.toolbox import filesio

if __name__ == '__main__':
    # VTK
    mi = ModelVTK('6623dd4fb5014a978d59b9acb03946d2')
    mi.flow_vtk()
    mi.transport_vtk()
    mi.conc_vtk()
    mi.particles_vtk(path=1)
    mi.wels_vtk()




