#  Copyright (c) 2020. Robin Thibaut, Ghent University

from experiment.goggles.vtkio import ModelVTK
from experiment.goggles.visualization import Plot
from experiment.toolbox import filesio

if __name__ == '__main__':
    # VTK
    mi = ModelVTK('0128284351704e91a8521cfc8c535df8')
    mi.flow_vtk()
    mi.transport_vtk()
    mi.conc_vtk()
    # mi.particles_vtk(path=0)
    mi.wels_vtk()




