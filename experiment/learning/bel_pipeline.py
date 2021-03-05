#  Copyright (c) 2021. Robin Thibaut, Ghent University
"""
This script pre-processes the data.

- It subdivides the breakthrough curves into an arbitrary number of steps, as the mt3dms results
do not necessarily share the same time steps - d

- It computes the signed distance field for each particles endpoints file - h
It then perform PCA keeping all components on both d and h.

- Finally, CCA is performed after selecting an appropriate number of PC to keep.

It saves 2 pca objects (d, h) and 1 cca object, according to the project ecosystem.
"""

import os
import shutil
import warnings
from os.path import join as jp
from typing import List

import joblib
import numpy as np

from .. import utils
from ..config import Setup

from ..design.forecast_error import Root, UncertaintyQuantification
from ..processing import predictor_handle as dops
import experiment.utils
from ..algorithms.cross_decomposition import CCA
from ..spatial import grid_parameters, signed_distance
from ..processing.dimension_reduction import PC

Root = List[str]
Combination = List[int]


def base_pca(
    base,
    base_dir: str,
    roots: Root,
    test_roots: Root,
    d_pca_obj=None,
    h_pca_obj=None,
):
    """
    Initiate BEL by performing PCA on the training targets or features.
    :param base: class: Base class object
    :param base_dir: str: Base directory path
    :param roots: list:
    :param test_roots: list:
    :param d_pca_obj:
    :param h_pca_obj:
    :return:
    """
    if d_pca_obj is not None:
        # Loads the results:
        tc0, _, _ = experiment.utils.data_loader(roots=roots, d=True)
        # tc0 = breakthrough curves with shape (n_sim, n_wells, n_time_steps)
        # pzs = WHPA
        # roots_ = simulation id
        # Subdivide d in an arbitrary number of time steps:
        # tc has shape (n_sim, n_wells, n_time_steps)
        tc = dops.curve_interpolation(tc0=tc0)
        # with n_sim = n_training + n_test
        # PCA on transport curves
        d_pco = PC(name="d",
                   training=tc,
                   roots=roots,
                   directory=os.path.dirname(d_pca_obj))
        d_pco.training_fit_transform()
        # Dump
        joblib.dump(d_pco, d_pca_obj)

    x_lim, y_lim, grf = base.Focus.x_range, base.Focus.y_range, base.Focus.cell_dim

    if h_pca_obj is not None:
        # Loads the results:
        _, pzs, r = experiment.utils.data_loader(roots=roots, h=True)
        # Load parameters:
        xys, nrow, ncol = grid_parameters(x_lim=x_lim, y_lim=y_lim,
                                          grf=grf)  # Initiate SD instance

        # PCA on signed distance
        # Compute signed distance on pzs.
        # h is the matrix of target feature on which PCA will be performed.
        h = np.array([signed_distance(xys, nrow, ncol, grf, pp) for pp in pzs])

        # Initiate h pca object
        h_pco = PC(name="h", training=h, roots=roots, directory=base_dir)
        # Transform
        h_pco.training_fit_transform()
        # Define number of components to keep
        # h_pco.n_pca_components(.98)  # Number of components for signed distance automatically set.
        h_pco.n_pc_cut = base.HyperParameters.n_pc_target
        # Dump
        joblib.dump(h_pco, h_pca_obj)

        # Save roots id's in a dat file
        if not os.path.exists(jp(base_dir, "roots.dat")):
            with open(jp(base_dir, "roots.dat"), "w") as f:
                for r in roots:  # Saves roots name until test roots
                    f.write(os.path.basename(r) + "\n")

        # Save roots id's in a dat file
        if not os.path.exists(jp(base_dir, "test_roots.dat")):
            with open(jp(base_dir, "test_roots.dat"), "w") as f:
                for r in test_roots:  # Saves roots name until test roots
                    f.write(os.path.basename(r) + "\n")


def bel_fit_transform(
    base,
    well_comb: Combination = None,
    training_roots: Root = None,
    test_root: Root = None,
):
    """
    This function loads raw data and perform both PCA and CCA on it.
    It saves results as pkl objects that have to be loaded in the forecast_error.py script to perform predictions.

    :param training_roots: list: List containing the uuid's of training roots
    :param base: class: Base class object containing global constants.
    :param well_comb: list: List of injection wells used to make prediction
    :param test_root: list: Folder path containing output to be predicted
    """

    # Load parameters:
    x_lim, y_lim, grf = base.Focus.x_range, base.Focus.y_range, base.Focus.cell_dim
    xys, nrow, ncol = grid_parameters(x_lim=x_lim, y_lim=y_lim,
                                      grf=grf)  # Initiate SD instance

    if well_comb is not None:
        base.Wells.combination = well_comb

    # Directories
    md = base.Directories
    res_dir = md.hydro_res_dir  # Results folders of the hydro simulations

    # Parse test_root
    if isinstance(test_root, str):  # If only one root given
        if os.path.exists(jp(res_dir, test_root)):
            test_root = [test_root]
        else:
            warnings.warn("Specified folder {} does not exist".format(
                test_root[0]))

    # Directory in which to load forecasts
    bel_dir = jp(md.forecasts_dir, test_root[0])

    # Base directory that will contain target objects and processed data
    base_dir = jp(md.forecasts_dir, "base")

    new_dir = "".join(list(map(
        str, base.Wells.combination)))  # sub-directory for forecasts
    sub_dir = jp(bel_dir, new_dir)

    # %% Folders
    obj_dir = jp(sub_dir, "obj")
    fig_data_dir = jp(sub_dir, "data")
    fig_pca_dir = jp(sub_dir, "pca")
    fig_cca_dir = jp(sub_dir, "cca")
    fig_pred_dir = jp(sub_dir, "uq")

    # %% Creates directories
    [
        experiment.utils.dirmaker(f) for f in
        [obj_dir, fig_data_dir, fig_pca_dir, fig_cca_dir, fig_pred_dir]
    ]

    # Load training data
    # Refined breakthrough curves data file
    tsub = jp(base_dir, "training_curves.npy")
    if not os.path.exists(tsub):
        # Loads the results:
        tc0, _, _ = experiment.utils.data_loader(res_dir=res_dir,
                                                 roots=training_roots,
                                                 d=True)
        # tc0 = breakthrough curves with shape (n_sim, n_wells, n_time_steps)
        # pzs = WHPA's
        # roots_ = simulations id's
        # Subdivide d in an arbitrary number of time steps:
        # tc has shape (n_sim, n_wells, n_time_steps)
        tc = dops.curve_interpolation(tc0=tc0, n_time_steps=200)
        # with n_sim = n_training + n_test
        np.save(tsub, tc)
        # Save file roots
    else:
        tc = np.load(tsub)

    # %% Select wells:
    selection = [wc - 1 for wc in base.Wells.combination]
    tc = tc[:, selection, :]

    # %%  PCA
    # PCA is performed with maximum number of components.
    # We choose an appropriate number of components to keep later on.

    # PCA on transport curves
    d_pco = PC(name="d", training=tc, roots=training_roots, directory=obj_dir)
    d_pco.training_fit_transform()
    # PCA on transport curves
    d_pco.n_pc_cut = base.HyperParameters.n_pc_predictor
    ndo = d_pco.n_pc_cut
    n_time_steps = base.HyperParameters.n_tstp
    # Load observation (test_root)
    tc0, _, _ = experiment.utils.data_loader(res_dir=res_dir,
                                             test_roots=test_root,
                                             d=True)
    # Subdivide d in an arbitrary number of time steps:
    tcp = dops.curve_interpolation(tc0=tc0, n_time_steps=n_time_steps)
    tcp = tcp[:, selection, :]  # Extract desired observation
    # Perform transformation on testing curves
    d_pco.test_transform(tcp, test_root=test_root)
    d_pc_training, _ = d_pco.comp_refresh(ndo)  # Split

    # Save the d PC object.
    joblib.dump(d_pco, jp(obj_dir, "d_pca.pkl"))

    # PCA on signed distance from base object containing training instances
    h_pco = joblib.load(jp(base_dir, "h_pca.pkl"))
    nho = h_pco.n_pc_cut  # Number of components to keep
    # Load whpa to predict
    _, pzs, _ = experiment.utils.data_loader(roots=test_root, h=True)
    # Compute WHPA on the prediction
    if h_pco.predict_pc is None:
        h = np.array([signed_distance(xys, nrow, ncol, grf, pp) for pp in pzs])
        # Perform PCA
        h_pco.test_transform(h, test_root=test_root)
        # Cut desired number of components
        h_pc_training, _ = h_pco.comp_refresh(nho)
        # Save updated PCA object in base
        joblib.dump(h_pco, jp(base_dir, "h_pca.pkl"))

        fig_dir = jp(base_dir, "roots_whpa")
        experiment.utils.dirmaker(fig_dir)
        np.save(jp(fig_dir, test_root[0]), h)  # Save the prediction WHPA
    else:
        # Cut components
        h_pc_training, _ = h_pco.comp_refresh(nho)

    # %% CCA
    # Number of CCA components is chosen as the min number of PC
    n_comp_cca = min(ndo, nho)
    # components between d and h.
    # By default, it scales the data
    # TODO: Check max_iter & tol
    cca = CCA(n_components=n_comp_cca,
              scale=True,
              max_iter=500 * 20,
              tol=1e-06)
    cca.fit(X=d_pc_training, Y=h_pc_training)  # Fit
    joblib.dump(cca, jp(obj_dir, "cca.pkl"))  # Save the fitted CCA operator

    return sub_dir


def scan_roots(base,
               training: Root,
               obs: Root,
               combinations: List[int],
               base_dir_path: str = None) -> float:
    """
    Scan forward roots and perform base decomposition.
    :param base: class: Base class (inventory)
    :param training: list: List of uuid of each root for training
    :param obs: list: List of uuid of each root for observation
    :param combinations: list: List of wells combinations, e.g. [[1, 2, 3, 4, 5, 6]]
    :param base_dir_path: str: Path to the base directory containing training roots uuid file
    :return: MHD mean (float)
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
    fb = utils.dirmaker(
        obj_path)  # Returns bool according to folder status
    if flag_base:
        utils.dirmaker(obj_path, erase=flag_base)
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
        belcomb = utils.combinator(
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