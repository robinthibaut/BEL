#  Copyright (c) 2020. Robin Thibaut, Ghent University

from bel.bel.forecast_error import UncertaintyQuantification

if __name__ == '__main__':
    uq = UncertaintyQuantification(study_folder='3d077529-0a0f-4fc5-9822-0a351503583e')
    for i in range(uq.n_test):
        uq.sample_posterior(sample_n=i)
        uq.c0(write_vtk=0)
        uq.hausdorff()
