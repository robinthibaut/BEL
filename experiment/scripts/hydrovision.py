#  Copyright (c) 2020. Robin Thibaut, Ghent University

from experiment.base.inventory import MySetup
from experiment.goggles.vtkio import ModelVTK
from experiment.goggles.visualization import Plot
from experiment.toolbox import filesio

if __name__ == '__main__':
    # VTK
    MySetup.Directories.hydro_res_dir = '/Users/robin/OneDrive - UGent/Project-we13c420/experiment/hydro/test'
    mi = ModelVTK(base=MySetup, folder='macos')
    mi.flow_vtk()
    # mi.transport_vtk()
    # mi.conc_vtk()
    # mi.particles_vtk(path=1)
    mi.wels_vtk()




