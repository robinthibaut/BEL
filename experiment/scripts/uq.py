#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import shutil
import itertools
import numpy as np
import matplotlib.pyplot as plt
from experiment.toolbox import filesio
from experiment.processing import decomposition as dcp
from experiment.bel.forecast_error import UncertaintyQuantification
from experiment.base.inventory import Directories, Wels


def combinator(combi):
    """Given a n-sized 1D array, generates all possible configurations, from size 1 to n-1.
    'None' will indicate to use the original combination.
    """
    cb = [list(itertools.combinations(combi, i)) for i in range(1, combi)]  # Get all possible wel combinations
    cb = [item for sublist in cb for item in sublist]  # Flatten and add None to first compute the
    # 'base'
    return cb


def scan_roots(training, obs, target_pca=None):
    """Scan all roots and perform base decomposition"""

    if not isinstance(obs, (list, tuple, np.array)):
        obs = [obs]

    if target_pca is not None:
        base_dir = target_pca
    else:
        base_dir = None

    # for r_ in obs:
    #     try:
    #
    #         sf = dcp.bel(training_roots=training, test_roots=r_, wel_comb=None, target_pca=target_pca)
    #
    #         uq = UncertaintyQuantification(study_folder=sf, base_dir=base_dir, wel_comb=None)
    #         uq.sample_posterior(sample_n=0, n_posts=500)  # Sample posterior
    #
    #     except Exception as ex:
    #         print(ex)

    wels = Wels()  # Load wels data from base
    comb = wels.combination  # Get default combination (all)
    belcomb = combinator(comb)

    for c in belcomb:
        try:
            sf = dcp.bel(training_roots=training, test_roots=obs, wel_comb=c, target_pca=target_pca)

            uq = UncertaintyQuantification(study_folder=sf, base_dir=base_dir, wel_comb=c)
            uq.sample_posterior(sample_n=0, n_posts=500)  # Sample posterior
            uq.c0(write_vtk=0)  # Extract 0 contours
            uq.mhd()  # Modified Hausdorff
            # uq.binary_stack()  # Binary stack
            # uq.kernel_density()  # Kernel density
        except Exception as ex:
            print(ex)


def value_info(root):

    droot = os.path.join(Directories.forecasts_dir, root)
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
    plt.title('Value of information for each wel')
    plt.xlabel('Wel ID')
    plt.xticks(np.arange(0, 7), wid)
    plt.ylabel('MHD summed mean value')
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(droot, 'wel_value.png'), dpi=300)
    plt.show()


if __name__ == '__main__':
    # TODO restructure BEL to have more control on root picking
    # TODO there is a strange WHPA in the lot
    md = Directories.hydro_res_dir

    roots_training = os.listdir(md)[:200]  # List of n training roots
    roots_obs = os.listdir(md)[200:300]  # List of m observation roots

    # Perform PCA on target (whpa) and store the object in a base folder
    obj_path = os.path.join(Directories.forecasts_dir, 'base_')
    filesio.dirmaker(obj_path)
    obj = os.path.join(obj_path, 'h_pca.pkl')
    # dcp.roots_pca(roots=roots_training, h_pca_obj=obj)

    # Perform base decomposition on the m roots
    scan_roots(training=roots_training, obs=roots_obs, target_pca=obj)







