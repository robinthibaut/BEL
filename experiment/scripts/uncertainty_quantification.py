#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
from typing import List

import joblib
import numpy as np

from experiment.base.inventory import MySetup
from experiment.processing import decomposition as dcp
from experiment.toolbox import filesio, utils
from experiment.uq.forecast_error import UncertaintyQuantification

Root = List[str]


def scan_roots(base,
               training: Root,
               obs: Root,
               combinations: List[int],
               base_dir_path: str = None):
    """
    Scan forward roots and perform base decomposition
    :param base: class: Base class (inventory)
    :param training: list: List of uuid of each root for training
    :param obs: list: List of uuid of each root for observation
    :param combinations: list: List of wells combinations, e.g. [[1, 2, 3, 4, 5, 6]]
    :param base_dir_path: str: Path to the base directory containing training roots uuid file
    :return:
    """

    if not isinstance(obs, (list, tuple)):
        obs = [obs]

    if not isinstance(combinations, (list, tuple)):
        combinations = [combinations]

    # Resets the target PCA object' predictions to None before starting
    try:
        joblib.load(os.path.join(base_dir_path, 'h_pca.pkl')).reset_()
    except FileNotFoundError:
        pass

    for r_ in obs:  # For each observation root
        for c in combinations:  # For each wel combination
            # PCA decomposition + CCA
            sf = dcp.bel(base=base, training_roots=training, test_root=r_, well_comb=c)
            # Uncertainty analysis
            uq = UncertaintyQuantification(base=base, study_folder=sf, base_dir=base_dir_path, wel_comb=c, seed=123456)
            uq.sample_posterior(n_posts=MySetup.Forecast.n_posts)  # Sample posterior
            uq.c0(write_vtk=False)  # Extract 0 contours
            uq.mhd()  # Modified Hausdorff
            # uq.binary_stack()
            # uq.kernel_density()

        # Resets the target PCA object' predictions to None before moving on to the next root
        joblib.load(os.path.join(base_dir_path, 'h_pca.pkl')).reset_()


def main(comb: List[List[int]] = None,
         n_training: int = 200,
         n_observations: int = 50,
         flag_base: bool = False,
         roots_training: Root = None,
         to_swap: Root = None,
         roots_obs: Root = None):
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

    def swap_root(pres: str):
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
               base_dir_path=obj_path)

    return roots_training, roots_obs


if __name__ == '__main__':
    # List directories in forwards folder
    base_dir = os.path.join(MySetup.Directories.forecasts_dir, 'base')

    training_roots = filesio.data_read(os.path.join(base_dir, 'roots.dat'))
    training_roots = [item for sublist in training_roots for item in sublist]

    test_roots = filesio.data_read(os.path.join(base_dir, 'test_roots.dat'))
    test_roots = [item for sublist in test_roots for item in sublist]

    # wells = [[1, 2, 3, 4, 5, 6], [1], [2], [3], [4], [5], [6]]
    wells = [[1, 2, 3, 4, 5, 6], [1], [2], [3], [4], [5], [6]]
    rt, ro = main(comb=wells,
                  roots_training=training_roots,
                  roots_obs=test_roots,
                  # n_training=200,
                  # n_observations=50,
                  flag_base=True)
