#  Copyright (c) 2020. Robin Thibaut, Ghent University

import shutil
from experiment.processing import decomposition as dcp
from experiment.bel.forecast_error import UncertaintyQuantification

if __name__ == '__main__':
    # try:
    #     sf = dcp.bel(test_roots='46933e56d83d4ddcaa26fa0cd8a795db')
    # except Exception as e:
    #     print(e)
    #     shutil.rmtree(sf)

    uq = UncertaintyQuantification(study_folder='271f557f-df78-47aa-90ad-20a949ea2a1b')
    uq.sample_posterior(sample_n=0, n_posts=500)  # Sample posterior
    uq.c0(write_vtk=0)  # Extract 0 contours
    mh = uq.mhd()  # Modified Hausdorff
    eb = uq.binary_stack()  # Binary stack
    uq.kernel_density()  # Kernel density
