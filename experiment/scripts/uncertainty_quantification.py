#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from experiment.toolbox import filesio, utils
from experiment.goggles.visualization import Plot
from experiment.processing import decomposition as dcp
from experiment.bel.forecast_error import UncertaintyQuantification
from experiment.base.inventory import MySetup


def scan_roots(base, training, obs, combinations, base_dir=None):
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
            sf = dcp.bel(base=base, training_roots=training, test_roots=r_, wel_comb=c)
            # Uncertainty analysis
            uq = UncertaintyQuantification(base=base, study_folder=sf, base_dir=base_dir, wel_comb=c)
            uq.sample_posterior(n_posts=MySetup.Forecast.n_posts, save_target_pc=True)  # Sample posterior
            uq.c0(write_vtk=0)  # Extract 0 contours
            uq.mhd()  # Modified Hausdorff
            # uq.binary_stack()
            # uq.kernel_density()

        # Resets the target PCA object' predictions to None before moving on to the next root
        joblib.load(os.path.join(base_dir, 'h_pca.pkl')).reset_()


def value_info(root):
    droot = os.path.join(MySetup.Directories.forecasts_dir, root)  # Starting point = root folder in forecast directory
    listit = os.listdir(droot)
    duq = list(filter(lambda f: os.path.isdir(os.path.join(droot, f)), listit))  # Folders of combinations

    wid = list(map(str, MySetup.Wels.combination))  # Wel identifiers (n)
    wm = np.zeros((len(wid), MySetup.Forecast.n_posts))  # Summed MHD when well i appears
    cid = np.copy(wm)  # Number of times each wel appears
    for e in duq:  # For each subfolder in the main folder
        fmhd = os.path.join(droot, e, 'uq', 'haus.npy')
        mhd = np.load(fmhd)  # Load MHD
        # mhd = np.mean(np.load(fmhd))  # Load MHD
        for w in wid:  # Check for each wel
            if w in e:  # If wel w is used
                idw = int(w) - 1  # -1 to respect 0 index
                wm[idw] += mhd  # Add mean of MHD
                cid[idw] += 1

    colors = Plot().cols

    # Plot
    # mode
    modes = []
    for i, m in enumerate(wm):
        count, values = np.histogram(m)
        idm = np.argmax(count)
        mode = values[idm]
        modes.append(mode)

    modes = np.array(modes)
    modes -= np.mean(modes)
    plt.bar(np.arange(1, 7), -modes, color=colors)
    plt.title('Value of information of each well')
    plt.xlabel('Well ID')
    plt.ylabel('Opposite deviation from mode\'s mean')
    plt.grid(color='#95a5a6', linestyle='--', linewidth=.5, axis='y', alpha=0.7)
    plt.savefig(os.path.join(MySetup.Directories.forecasts_dir, f'{root}_well_mode.png'), dpi=300, transparent=True)
    plt.show()

    # Plot histogram
    for i, m in enumerate(wm):
        sns.kdeplot(m, color=f'{colors[i]}', shade=True, linewidth=2)
    plt.title('Summed MHD distribution for each well')
    plt.xlabel('Summed MHD')
    plt.ylabel('KDE')
    plt.legend(wid)
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(MySetup.Directories.forecasts_dir, f'{root}_hist.png'), dpi=300, transparent=True)
    plt.show()


def main(flag_base=False, swap=False):
    """
    I. First, defines the roots for training from simulations in the hydro directory.
    II. Define one 'observation' root.
    III. Perform PCA decomposition on the training targets and store the otput in the 'base' folder,
    as to avoid recomputing it every time.
    IV. Given n combinations of data source, apply BEL approach n times and perform uncertainty quantification
    """

    if os.uname().nodename == 'MacBook-Pro.local':
        MySetup.Directories.hydro_res_dir = '/Users/robin/OneDrive - UGent/Project-we13c420/experiment/hydro/results'
        root_file = '/Users/robin/OneDrive - UGent/Project/experiment/bel/forecasts/base/roots.dat'
    else:
        # Reads in root files (contains uuid of training roots)
        root_file = os.path.join(MySetup.Directories.forecasts_dir, 'base', 'roots.dat')

    def swap_root(pres):
        """Selects roots from main folder and swap if necessary"""
        md = MySetup.Directories.hydro_res_dir
        listme = os.listdir(md)
        folders = list(filter(lambda f: os.path.isdir(os.path.join(md, f)), listme))
        r_training = folders[:200]  # List of n training roots
        r_obs = folders[200:300]  # List of m observation roots

        idx = r_training.index(pres)
        r_obs[0], r_training[idx] = r_training[idx], r_obs[0]

    if swap:
        swap_root('6623dd4fb5014a978d59b9acb03946d2')

    # root_file gives in a list of lists, flatten it:
    roots_training = [item for sublist in filesio.datread(root_file) for item in sublist]
    # Specify 'observation' root:
    roots_obs = ['6623dd4fb5014a978d59b9acb03946d2']

    # Perform PCA on target (whpa) and store the object in a base folder
    obj_path = os.path.join(MySetup.Directories.forecasts_dir, 'base')
    fb = filesio.dirmaker(obj_path)
    if flag_base and fb:
        # Creates main target PCA object
        obj = os.path.join(obj_path, 'h_pca.pkl')
        dcp.base_pca(base=MySetup, roots=roots_training, h_pca_obj=obj, check=False)

    comb = MySetup.Wels.combination  # Get default combination (all)
    belcomb = utils.combinator(comb)  # Get all possible combinations

    # Perform base decomposition on the m roots
    scan_roots(base=MySetup, training=roots_training, obs=roots_obs, combinations=belcomb, base_dir=obj_path)


if __name__ == '__main__':
    # main()
    value_info('6623dd4fb5014a978d59b9acb03946d2')
