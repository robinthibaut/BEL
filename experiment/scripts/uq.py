#  Copyright (c) 2020. Robin Thibaut, Ghent University

import shutil
import itertools
from experiment.processing import decomposition as dcp
from experiment.bel.forecast_error import UncertaintyQuantification
from experiment.base.inventory import Wels


if __name__ == '__main__':

    wels = Wels()
    comb = wels.combination
    belcomb = [list(itertools.combinations(comb, i)) for i in range(2, comb[-1])]
    belcomb = [item for sublist in belcomb for item in sublist]

    try:
        sf = dcp.bel(test_roots='46933e56d83d4ddcaa26fa0cd8a795db', wel_comb=belcomb)
    except Exception as e:
        print(e)
        shutil.rmtree(sf)

    uq = UncertaintyQuantification(study_folder=sf, wel_comb=belcomb)
    uq.sample_posterior(sample_n=0, n_posts=500)  # Sample posterior
    uq.c0(write_vtk=0)  # Extract 0 contours
    mh = uq.mhd()  # Modified Hausdorff
    eb = uq.binary_stack()  # Binary stack
    uq.kernel_density()  # Kernel density
