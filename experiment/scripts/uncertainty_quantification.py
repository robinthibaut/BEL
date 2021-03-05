#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
import shutil
from typing import List

import joblib
import numpy as np

import experiment.utils
from experiment.config import Setup
from experiment.design.forecast_error import UncertaintyQuantification
from experiment.learning import bel_pipeline as dcp

Root = List[str]


def scan_roots(base,
               training: Root,
               obs: Root,
               combinations: List[int],
               base_dir_path: str = None) -> float:
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
        joblib.load(os.path.join(base_dir_path, "h_pca.pkl")).reset_()
    except FileNotFoundError:
        pass

    global_mean = 0
    for r_ in obs:  # For each observation root
        for c in combinations:  # For each wel combination
            # PCA decomposition + CCA
            sf = dcp.bel_fit_transform(base=base,
                                       training_roots=training,
                                       test_root=r_,
                                       well_comb=c)
            # Uncertainty analysis
            uq = UncertaintyQuantification(
                base=base,
                study_folder=sf,
                base_dir=base_dir_path,
                wel_comb=c,
                seed=123456,
            )
            # Sample posterior
            uq.sample_posterior(n_posts=base.HyperParameters.n_posts)
            uq.c0(write_vtk=False)  # Extract 0 contours
            mean = uq.mhd()  # Modified Hausdorff
            global_mean += mean

        # Resets the target PCA object' predictions to None before moving on to the next root
        joblib.load(os.path.join(base_dir_path, "h_pca.pkl")).reset_()

    return global_mean


def analysis(
    base,
    comb: List[List[int]] = None,
    n_training: int = 200,
    n_obs: int = 50,
    flag_base: bool = False,
    wipe: bool = False,
    roots_training: Root = None,
    to_swap: Root = None,
    roots_obs: Root = None,
):
    """

    I. First, defines the roots for training from simulations in the hydro results directory.
    II. Define one 'observation' root (roots_obs in params).
    III. Perform PCA decomposition on the training targets and store the output in the 'base' folder,
    to avoid recomputing it every time.
    IV. Given n combinations of data source, apply BEL approach n times and perform uncertainty quantification.

    :param base: class: Base class (inventory)
    :param wipe: bool: Whether to wipe the 'forecast' folder or not
    :param comb: list: List of well IDs
    :param n_training: int: Index from which training and data are separated
    :param n_obs: int: Number of predictors to take
    :param flag_base: bool: Recompute base PCA on target if True
    :param roots_training: list: List of roots considered as training.
    :param to_swap: list: List of roots to swap from training to observations.
    :param roots_obs: list: List of roots considered as observations.
    :return: list: List of training roots, list: List of observation roots

    """
    # Results location
    md = base.Directories.hydro_res_dir
    listme = os.listdir(md)
    # Filter folders out
    folders = list(filter(lambda f: os.path.isdir(os.path.join(md, f)),
                          listme))

    def swap_root(pres: str):
        """Selects roots from main folder and swap them from training to observation"""
        if pres in roots_training:
            idx = roots_training.index(pres)
            roots_obs[0], roots_training[idx] = roots_training[idx], roots_obs[
                0]
        elif pres in folders:
            idx = folders.index(pres)
            roots_obs[0] = folders[idx]
        else:
            pass

    if roots_training is None:
        roots_training = folders[:n_training]  # List of n training roots
    else:
        n_training = len(roots_training)

    # base.HyperParameters.n_posts = n_training

    if roots_obs is None:  # If no observation provided
        if n_training + n_obs <= len(folders):
            # List of m observation roots
            roots_obs = folders[n_training:(n_training + n_obs)]
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
            print(f"obs {r} is located in the training roots")
            return

    if to_swap is not None:
        [swap_root(ts) for ts in to_swap]

    # Perform PCA on target (whpa) and store the object in a base folder
    if wipe:
        try:
            shutil.rmtree(base.Directories.forecasts_dir)
        except FileNotFoundError:
            pass
    obj_path = os.path.join(base.Directories.forecasts_dir, "base")
    fb = experiment.utils.dirmaker(
        obj_path)  # Returns bool according to folder status
    if flag_base and not fb:
        experiment.utils.dirmaker(obj_path)
        # Creates main target PCA object
        obj = os.path.join(obj_path, "h_pca.pkl")
        dcp.base_pca(
            base=base,
            base_dir=obj_path,
            roots=roots_training,
            test_roots=roots_obs,
            h_pca_obj=obj
        )

    if comb is None:
        comb = base.Wells.combination  # Get default combination (all)
        belcomb = experiment.utils.combinator(
            comb)  # Get all possible combinations
    else:
        belcomb = comb

    # Perform base decomposition on the m roots
    global_mean = scan_roots(
        base=base,
        training=roots_training,
        obs=roots_obs,
        combinations=belcomb,
        base_dir_path=obj_path,
    )

    if wipe:
        shutil.rmtree(Setup.Directories.forecasts_dir)

    return roots_training, roots_obs, global_mean


def get_roots(training_file: str = None,
              test_file: str = None):

    if training_file is None:
        training_file = os.path.join(Setup.Directories.storage_dir, "roots.dat")
    if test_file is None:
        test_file = os.path.join(Setup.Directories.storage_dir, "test_roots.dat")

    # List directories in forwards folder
    training_roots = experiment.utils.data_read(training_file)
    training_roots = [item for sublist in training_roots for item in sublist]

    test_roots = experiment.utils.data_read(test_file)
    test_roots = [item for sublist in test_roots for item in sublist]

    return training_roots, test_roots


def main_1():
    training_r, test_r = get_roots()

    # wells = [[1, 2, 3, 4, 5, 6], [1], [2], [3], [4], [5], [6]]
    wells = [[1, 2, 3, 4, 5, 6]]
    analysis(
        base=Setup,
        comb=wells,
        roots_training=training_r,
        roots_obs=test_r,
        wipe=False,
        flag_base=True,
    )


def main_2(N):
    means = []

    for n in N:
        Setup.HyperParameters.n_total = n
        Setup.HyperParameters.n_training = int(n * 0.8)
        print(f"n_training={int(n * .8)}")
        Setup.HyperParameters.n_test = int(n * 0.2)
        print(f"n_test={int(n * .2)}")

        # wells = [[1, 2, 3, 4, 5, 6], [1], [2], [3], [4], [5], [6]]
        wells = [[1, 2, 3, 4, 5, 6]]
        *_, mhd_mean = analysis(
            base=Setup,
            comb=wells,
            n_training=Setup.HyperParameters.n_training,
            n_obs=Setup.HyperParameters.n_test,
            wipe=True,
            flag_base=True,
        )
        means.append([n, mhd_mean])

    return np.array(means)


def test():
    training_file = os.path.join(Setup.Directories.storage_dir, "test", "roots.dat")
    test_file = os.path.join(Setup.Directories.storage_dir, "test", "test_roots.dat")
    training_r, test_r = get_roots(training_file=training_file,
                                   test_file=test_file)

    wells = [[1, 2, 3, 4, 5, 6]]

    test_base = Setup

    test_base.Directories.forecasts_dir = \
        os.path.join(test_base.Directories.storage_dir, "test")

    analysis(
        base=test_base,
        comb=wells,
        roots_training=training_r,
        roots_obs=test_r,
        wipe=False,
        flag_base=True,
    )


if __name__ == "__main__":
    main_1()
    # n_try = np.linspace(100, 2000, 50)
    # n_try = [100]
    # mv = main_2(N=n_try)
    # np.save(os.path.join(Setup.Directories.storage_dir, "means.npy"), mv)
