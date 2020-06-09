#  Copyright (c) 2020. Robin Thibaut, Ghent University

import shutil
import itertools
from experiment.processing import decomposition as dcp
from experiment.bel.forecast_error import UncertaintyQuantification
from experiment.base.inventory import Wels


if __name__ == '__main__':

    wels = Wels()  # Load wels data from base
    comb = wels.combination  # Get default combination (all)
    belcomb = [list(itertools.combinations(comb, i)) for i in range(2, comb[-1])]  # Get all possible wel combinations
    belcomb = [None] + [item for sublist in belcomb for item in sublist]  # Flatten and add None to first compute the
    # 'base'

    for c in belcomb:
        try:
            sf = dcp.bel(test_roots='46933e56d83d4ddcaa26fa0cd8a795db', wel_comb=c, base=1)
        except Exception as e:
            print(e)
            shutil.rmtree(sf)

        uq = UncertaintyQuantification(study_folder=sf, wel_comb=c)
        uq.sample_posterior(sample_n=0, n_posts=500)  # Sample posterior
        uq.c0(write_vtk=0)  # Extract 0 contours
        mh = uq.mhd()  # Modified Hausdorff
        eb = uq.binary_stack()  # Binary stack
        uq.kernel_density()  # Kernel density
