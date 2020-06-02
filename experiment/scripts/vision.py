#  Copyright (c) 2020. Robin Thibaut, Ghent University

from experiment.goggles.vtkio import ModelVTK

if __name__ == '__main__':
    mi = ModelVTK('simulation')
    mi.flow_vtk()
    mi.transport_vtk()
    mi.particles_vtk(path=0)
    mi.wels_vtk()
