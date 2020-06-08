#  Copyright (c) 2020. Robin Thibaut, Ghent University

from experiment.goggles.vtkio import ModelVTK
from experiment.goggles.visualization import Plot
from experiment.toolbox import filesio

if __name__ == '__main__':
    # VTK
    # mi = ModelVTK('simulation')
    # mi.flow_vtk()
    # mi.transport_vtk()
    # mi.particles_vtk(path=0)
    # mi.wels_vtk()
    myplot = Plot()

    myplot.check_root(root='46933e56d83d4ddcaa26fa0cd8a795db')



