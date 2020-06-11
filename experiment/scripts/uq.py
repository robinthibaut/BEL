#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import shutil
import itertools
from experiment.processing import decomposition as dcp
from experiment.bel.forecast_error import UncertaintyQuantification
from experiment.base.inventory import Directories, Wels


def combinator(combi):
    cb = [list(itertools.combinations(combi, i)) for i in range(1, combi[-1])]  # Get all possible wel combinations
    cb = [None] + [item for sublist in cb for item in sublist]  # Flatten and add None to first compute the
    # 'base'
    return cb


def scan_root(root):
    wels = Wels()  # Load wels data from base
    comb = wels.combination  # Get default combination (all)
    belcomb = [list(itertools.combinations(comb, i)) for i in range(1, comb[-1])]  # Get all possible wel combinations
    belcomb = [None] + [item for sublist in belcomb for item in sublist]  # Flatten and add None to first compute the
    # 'base'

    for c in belcomb:
        try:
            sf = dcp.bel(test_roots=root, wel_comb=c, base=1)
        except Exception as e:
            print(e)
            shutil.rmtree(sf)

        uq = UncertaintyQuantification(study_folder=sf, wel_comb=c)
        uq.sample_posterior(sample_n=0, n_posts=500)  # Sample posterior
        uq.c0(write_vtk=0)  # Extract 0 contours
        mh = uq.mhd()  # Modified Hausdorff
        # eb = uq.binary_stack()  # Binary stack
        # uq.kernel_density()  # Kernel density


def scan_roots():
    wels = Wels()  # Load wels data from base
    comb = wels.combination  # Get default combination (all)

    belcomb = combinator(comb)

    md = Directories.hydro_res_dir
    roots = os.listdir(md)

    for r in roots[:10]:
        try:

            sf = dcp.bel(test_roots=r, wel_comb=None, base=1)

            uq = UncertaintyQuantification(study_folder=sf, wel_comb=None)
            uq.sample_posterior(sample_n=0, n_posts=500)  # Sample posterior
            uq.c0(write_vtk=0)  # Extract 0 contours
            mh = uq.mhd()  # Modified Hausdorff
            # eb = uq.binary_stack()  # Binary stack
            # uq.kernel_density()  # Kernel density
        except Exception as e:
            print(e)


if __name__ == '__main__':
    scan_roots()

