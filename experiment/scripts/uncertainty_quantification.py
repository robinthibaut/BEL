#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiment.base.inventory import MySetup
from experiment.uq.forecast_error import UncertaintyQuantification
from experiment.goggles.visualization import Plot
from experiment.processing import decomposition as dcp
from experiment.toolbox import filesio, utils


def value_info(root):
    """
    Computes the combined value of information for n observations.
    :param root: list: List containing the roots whose wells contributions will be taken into account.
    :return:
    """
    if not isinstance(root, (list, tuple)):
        root = [root]

    wid = list(map(str, MySetup.Wells.combination))  # Wel identifiers (n)
    wm = np.zeros((len(wid), MySetup.Forecast.n_posts))  # Summed MHD when well i appears

    for r in root:  # For each root
        droot = os.path.join(MySetup.Directories.forecasts_dir, r)  # Starting point = root folder in forecast directory
        for e in wid:  # For each subfolder (well) in the main folder
            fmhd = os.path.join(droot, e, 'obj', 'haus.npy')  # Get the MHD file
            mhd = np.load(fmhd)  # Load MHD
            idw = int(e) - 1  # -1 to respect 0 index (Well index)
            wm[idw] += mhd  # Add MHD at each well

    colors = Plot().cols  # Get default colors from visualization class

    modes = []  # Get MHD corresponding to each well's mode
    for i, m in enumerate(wm):  # For each well, look up its MHD distribution
        count, values = np.histogram(m)
        idm = np.argmax(count)
        mode = values[idm]
        modes.append(mode)

    # TODO: Put visualization methods in proper folder
    modes = np.array(modes)  # Scale modes
    modes -= np.mean(modes)

    # Bar plot
    plt.bar(np.arange(1, 7), -modes, color=colors)
    plt.title('Value of information of each well')
    plt.xlabel('Well ID')
    plt.ylabel('Opposite deviation from mode\'s mean')
    plt.grid(color='#95a5a6', linestyle='--', linewidth=.5, axis='y', alpha=0.7)
    plt.savefig(os.path.join(MySetup.Directories.forecasts_dir, 'well_mode.png'), dpi=300, transparent=True)
    plt.close()
    # plt.show()

    # Plot histogram
    for i, m in enumerate(wm):
        sns.kdeplot(m, color=f'{colors[i]}', shade=True, linewidth=2)
    plt.title('Summed MHD distribution for each well')
    plt.xlabel('Summed MHD')
    plt.ylabel('KDE')
    plt.legend(wid)
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(MySetup.Directories.forecasts_dir, 'hist.png'), dpi=300, transparent=True)
    plt.close()
    # plt.show()

    # %% Facet histograms
    ids = np.array(np.concatenate([np.ones(wm.shape[1]) * i for i in range(1, 7)]), dtype='int')
    master = wm.flatten()

    data = np.concatenate([[master], [ids]], axis=0)

    master_x = pd.DataFrame(data=data.T, columns=['MHD', 'well'])
    master_x['well'] = np.array(ids)
    g = sns.FacetGrid(master_x,  # the dataframe to pull from
                      row="well",
                      hue="well",
                      aspect=3,  # aspect * height = width
                      height=1.5,  # height of each subplot
                      palette=colors  # google colors
                      )

    g.map(sns.kdeplot, "MHD", shade=True, alpha=1, lw=1.5)
    g.map(plt.axhline, y=0, lw=4)
    for ax in g.axes:
        ax[0].set_xlim((500, 1000))

    def label(x, color, label):
        ax = plt.gca()  # get the axes of the current object
        ax.text(0, .2,  # location of text
                label,  # text label
                fontweight="bold", color=color, size=20,  # text attributes
                ha="left", va="center",  # alignment specifications
                transform=ax.transAxes)  # specify axes of transformation)

    g.map(label, "MHD")  # the function counts as a plotting object!

    sns.set(style="dark", rc={"axes.facecolor": (0, 0, 0, 0)})
    g.fig.subplots_adjust(hspace=-.25)

    g.set_titles("")  # set title to blank
    g.set_xlabels(color="white")
    g.set_xticklabels(color='white', fontsize=14)
    g.set(yticks=[])  # set y ticks to blank
    g.despine(bottom=True, left=True)  # remove 'spines'
    plt.savefig(os.path.join(MySetup.Directories.forecasts_dir, 'facet.png'), dpi=300, transparent=True)
    # plt.show()


def scan_roots(base, training, obs, combinations, base_dir=None):
    """
    Scan forward roots and perform base decomposition
    :param base: class: Base class (inventory)
    :param training: list: List of uuid of each root for training
    :param obs: list: List of uuid of each root for observation
    :param combinations: list: List of wells combinations, e.g. [[1, 2, 3, 4, 5, 6]]
    :param base_dir: str: Path to the base directory containing training roots uuid file
    :return:
    """

    if not isinstance(obs, (list, tuple)):
        obs = [obs]

    if not isinstance(combinations, (list, tuple)):
        combinations = [combinations]

    # Resets the target PCA object' predictions to None before starting
    try:
        joblib.load(os.path.join(base_dir, 'h_pca.pkl')).reset_()
    except FileNotFoundError:
        pass

    for r_ in obs:  # For each observation root
        for c in combinations:  # For each wel combination
            # PCA decomposition + CCA
            sf = dcp.bel(base=base, training_roots=training, test_root=r_, wel_comb=c)
            # Uncertainty analysis
            uq = UncertaintyQuantification(base=base, study_folder=sf, base_dir=base_dir, wel_comb=c, seed=123456)
            uq.sample_posterior(n_posts=MySetup.Forecast.n_posts)  # Sample posterior
            uq.c0(write_vtk=0)  # Extract 0 contours
            uq.mhd()  # Modified Hausdorff
            # uq.binary_stack()
            # uq.kernel_density()

        # Resets the target PCA object' predictions to None before moving on to the next root
        joblib.load(os.path.join(base_dir, 'h_pca.pkl')).reset_()


def main(comb: list = None,
         n_training: int = 200,
         n_observations: int = 50,
         flag_base=False,
         roots_training=None,
         to_swap=None,
         roots_obs=None):
    """

    I. First, defines the roots for training from simulations in the hydro results directory.
    II. Define one 'observation' root (roots_obs in params).
    III. Perform PCA decomposition on the training targets and store the output in the 'base' folder,
    to avoid recomputing it every time.
    IV. Given n combinations of data source, apply BEL approach n times and perform uncertainty quantification.

    :param comb: list: List of well IDs
    :param n_training: int: Index from which training and data are separated
    :param n_observations: int: Number of predictors to take
    :param flag_base: bool: Recompute base PCA on target if True
    :param roots_training: list: List of roots considered as training.
    :param to_swap: list: List of roots to swap from training to observations.
    :param roots_obs: list: List of roots considered as observations.
    :return: list: List of training roots, list: List of observation roots

    """
    # Results location
    md = MySetup.Directories.hydro_res_dir
    listme = os.listdir(md)
    # Filter folders out
    folders = list(filter(lambda f: os.path.isdir(os.path.join(md, f)), listme))

    def swap_root(pres):
        """Selects roots from main folder and swap them from training to observation"""
        if pres in roots_training:
            idx = roots_training.index(pres)
            roots_obs[0], roots_training[idx] = roots_training[idx], roots_obs[0]
        elif pres in folders:
            idx = folders.index(pres)
            roots_obs[0] = folders[idx]
        else:
            pass

    if roots_training is None:
        roots_training = folders[:n_training]  # List of n training roots
    else:
        n_training = len(roots_training)

    MySetup.Forecast.n_posts = n_training

    if roots_obs is None:  # If no observation provided
        if n_training + n_observations <= len(folders):
            roots_obs = folders[n_training:(n_training + n_observations)]  # List of m observation roots
        else:
            print("Incompatible training/observation numbers")
            return

    for i, r in enumerate(roots_training):
        choices = folders[n_training:].copy()
        if r in roots_obs:
            random_root = np.random.choice(choices)
            roots_training[i] = random_root
            choices.remove(random_root)

    for r in roots_obs:
        if r in roots_training:
            print(f'obs {r} is located in the training roots')
            return

    if to_swap is not None:
        [swap_root(ts) for ts in to_swap]

    # Perform PCA on target (whpa) and store the object in a base folder
    obj_path = os.path.join(MySetup.Directories.forecasts_dir, 'base')
    fb = filesio.dirmaker(obj_path)  # Returns bool according to folder status
    if flag_base:
        # Creates main target PCA object
        obj = os.path.join(obj_path, 'h_pca.pkl')
        dcp.base_pca(base=MySetup,
                     base_dir=obj_path,
                     roots=roots_training,
                     test_roots=roots_obs,
                     h_pca_obj=obj,
                     check=False)

    if comb is None:
        comb = MySetup.Wells.combination  # Get default combination (all)
        belcomb = utils.combinator(comb)  # Get all possible combinations
    else:
        belcomb = comb

    # Perform base decomposition on the m roots
    scan_roots(base=MySetup,
               training=roots_training,
               obs=roots_obs,
               combinations=belcomb,
               base_dir=obj_path)

    return roots_training, roots_obs


if __name__ == '__main__':
    # List directories in forwards folder
    base_dir = os.path.join(MySetup.Directories.forecasts_dir, 'base')

    # listme = os.listdir(MySetup.Directories.hydro_res_dir)
    # pot_obs = [f for f in listme if os.path.exists(
    #     os.path.join(MySetup.Directories.hydro_res_dir, f, f'{MySetup.Directories.project_name}.hds'))]

    # training_roots = filesio.datread(os.path.join(MySetup.Directories.forecasts_dir, 'base', 'roots.dat'))
    # training_roots = [item for sublist in training_roots for item in sublist]

    # test_roots = filesio.datread(os.path.join(base_dir, 'test_roots.dat'))
    # test_roots = [item for sublist in test_roots for item in sublist]

    # wells = [[1, 2, 3, 4, 5, 6], [1], [2], [3], [4], [5], [6]]
    # wells = [[1, 2, 3, 4, 5, 6], [1], [2], [3], [4], [5], [6]]
    # rt, ro = main(comb=wells,
    #               n_training=300,
    #               n_observations=90,
    #               flag_base=True)
    # Value info
    forecast_dir = MySetup.Directories.forecasts_dir
    listit = os.listdir(forecast_dir)
    listit.remove('base')
    duq = list(filter(lambda f: os.path.isdir(os.path.join(forecast_dir, f)), listit))  # Folders of combinations
    value_info(duq)
