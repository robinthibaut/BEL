#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import joblib
import itertools
import numpy as np
import matplotlib.pyplot as plt
from experiment.toolbox import filesio
from experiment.goggles.visualization import Plot
from experiment.processing import decomposition as dcp
from experiment.bel.forecast_error import UncertaintyQuantification
from experiment.base.inventory import Directories, Wels


def combinator(combi):
    """Given a n-sized 1D array, generates all possible configurations, from size 1 to n-1.
    'None' will indicate to use the original combination.
    """
    cb = [list(itertools.combinations(combi, i)) for i in range(1, combi[-1]+1)]  # Get all possible wel combinations
    cb = [item for sublist in cb for item in sublist][::-1]  # Flatten and add None to first compute the
    # 'base'
    return cb


def scan_roots(training, obs, combinations, base_dir=None):
    """Scan all roots and perform base decomposition"""

    if not isinstance(obs, (list, tuple, np.array)):
        obs = [obs]

    if base_dir is not None:
        base_dir = base_dir
    else:
        base_dir = None

    # Resets the target PCA object' predictions to None before starting
    joblib.load(os.path.join(base_dir, 'h_pca.pkl')).reset_()

    for r_ in obs:  # For each observation root
        for c in combinations:  # For each wel combination
            # PCA decomposition + CCA
            sf = dcp.bel(training_roots=training, test_roots=r_, wel_comb=c)
            # Uncertainty analysis
            uq = UncertaintyQuantification(study_folder=sf, base_dir=base_dir, wel_comb=c)
            # uq.control()  # Compare PCA recoveries
            uq.sample_posterior(n_posts=500)  # Sample posterior
            uq.c0(write_vtk=0)  # Extract 0 contours
            uq.mhd()  # Modified Hausdorff
            uq.binary_stack()
            uq.kernel_density()

        # Resets the target PCA object' predictions to None before moving on to the next root
        joblib.load(os.path.join(base_dir, 'h_pca.pkl')).reset_()


def value_info(root):

    droot = os.path.join(Directories.forecasts_dir, root)
    duq = os.listdir(droot)  # Folders of combinations
    wid = list(map(str, Wels.combination))  # Wel identifiers (n)
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

    colors = Plot().cols
    for i, m in enumerate(wm):
        plt.plot(i, m, f'{colors[i]}o')
    plt.title('Value of information for each well')
    plt.xlabel('Well ID')
    plt.xticks(np.arange(0, 7), wid)
    plt.ylabel('MHD summed mean value')
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(Directories.forecasts_dir, f'{root}_well_value.png'), dpi=300)
    plt.show()


def main():
    md = Directories.hydro_res_dir

    roots_training = os.listdir(md)[:200]  # List of n training roots
    roots_obs = os.listdir(md)[200:300]  # List of m observation roots

    pres = '6623dd4fb5014a978d59b9acb03946d2'
    idx = roots_training.index(pres)
    roots_obs[0], roots_training[idx] = roots_training[idx], roots_obs[0]

    # Perform PCA on target (whpa) and store the object in a base folder
    obj_path = os.path.join(Directories.forecasts_dir, 'base')
    filesio.dirmaker(obj_path)
    obj = os.path.join(obj_path, 'h_pca.pkl')
    dcp.base_pca(roots=roots_training, h_pca_obj=obj, check=False)

    comb = Wels.combination  # Get default combination (all)
    belcomb = combinator(comb)  # Get all possible combinations

    # sa = belcomb.index((5, 6))
    # belcomb = belcomb[sa:]

    # Perform base decomposition on the m roots
    scan_roots(training=roots_training, obs=[roots_obs[0]], combinations=belcomb, base_dir=obj_path)


if __name__ == '__main__':
    value_info('6623dd4fb5014a978d59b9acb03946d2')









