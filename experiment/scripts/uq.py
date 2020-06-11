#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import shutil
import itertools
import numpy as np
import matplotlib.pyplot as plt
from experiment.processing import decomposition as dcp
from experiment.bel.forecast_error import UncertaintyQuantification
from experiment.base.inventory import Directories, Wels


def combinator(combi):
    """Given a n-sized 1D array, generates all possible configurations, from size 1 to n-1.
    'None' will indicate to use the original combination.
    """
    cb = [list(itertools.combinations(combi, i)) for i in range(1, combi[-1])]  # Get all possible wel combinations
    cb = [None] + [item for sublist in cb for item in sublist]  # Flatten and add None to first compute the
    # 'base'
    return cb


def scan_root(root):
    """Takes a root name and performs decomposition and UQ with all wels combinations"""
    wels = Wels()  # Load wels data from base
    comb = wels.combination  # Get default combination (all)
    belcomb = [list(itertools.combinations(comb, i)) for i in range(1, comb[-1])]  # Get all possible wel combinations
    belcomb = [None] + [item for sublist in belcomb for item in sublist]  # Flatten and add None to first compute the
    # 'base'

    for c in belcomb:
        try:
            sf = dcp.bel(test_roots=root, wel_comb=c, base=1)

            uq = UncertaintyQuantification(study_folder=sf, wel_comb=c)
            uq.sample_posterior(sample_n=0, n_posts=500)  # Sample posterior
            uq.c0(write_vtk=0)  # Extract 0 contours
            uq.mhd()  # Modified Hausdorff
            uq.binary_stack()  # Binary stack
            # uq.kernel_density()  # Kernel density
        except Exception as ex:
            print(ex)


def scan_roots():
    """Scan all roots and perform decomposition"""
    md = Directories.hydro_res_dir
    roots = os.listdir(md)  # List all roots

    for r in roots[:10]:
        try:

            sf = dcp.bel(test_roots=r, wel_comb=None, base=1)

            uq = UncertaintyQuantification(study_folder=sf, wel_comb=None)
            uq.sample_posterior(sample_n=0, n_posts=500)  # Sample posterior
            # uq.c0(write_vtk=0)  # Extract 0 contours
            # mh = uq.mhd()  # Modified Hausdorff
            # eb = uq.binary_stack()  # Binary stack
            # uq.kernel_density()  # Kernel density
        except Exception as ex:
            print(ex)


if __name__ == '__main__':
    # scan_root('0128284351704e91a8521cfc8c535df8')

    droot = os.path.join(Directories.forecasts_dir, '0128284351704e91a8521cfc8c535df8')
    duq = os.listdir(droot)  # Folders of combinations
    wid = list(map(str, range(1, 7)))  # Wel identifiers (n)
    wm = np.zeros(len(wid))  # Summed MHD when well i appears
    cid = np.copy(wm)  # Number of times each wel appears
    for e in duq:
        fmhd = os.path.join(droot, e, 'uq', 'haus.npy')
        mhd = np.mean(np.load(fmhd))  # Load MHD
        for w in wid:  # Check for each wel
            if w in e:  # If wel w is used
                idw = int(w)-1
                wm[idw] += mhd  # Add mean of MHD
                cid[idw] += 1

    plt.plot(wm, 'wo')
    plt.title('Summed MHD for each wel')
    plt.xlabel('Wel ID')
    plt.xticks(np.arange(0, 7), wid)
    plt.ylabel('MHD summed mean value')
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(droot, 'wel_value.png'), dpi=300)
    plt.show()





