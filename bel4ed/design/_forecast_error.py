#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
import shutil
from os.path import join as jp
from typing import List, Type

import joblib
import numpy as np
import vtk
from loguru import logger

from .. import utils
from ..config import Setup
from ..utils import Root, Function
from ..learning.bel_pipeline import bel_fit_transform, base_pca, PosteriorIO
from ..spatial import (
    contours_vertices,
    refine_machine,
)

__all__ = ['UncertaintyQuantification', 'analysis', 'measure_info_mode']


class UncertaintyQuantification:
    def __init__(
            self,
            base: Type[Setup],
            study_folder: str,
            base_dir: str = None,
            seed: int = None,
    ):
        """

        :param base: class: Base object (inventory)
        :param study_folder: str: Name of the root uuid in the 'forecast' directory on which UQ will be performed
        :param base_dir: str: Path to base directory
        :param seed: int: Seed
        """

        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None

        self.base = base

        fc = self.base.Focus
        self.x_lim, self.y_lim, self.grf = fc.x_range, fc.y_range, fc.cell_dim

        # Directories & files paths
        md = self.base.Directories
        self.main_dir = md.main_dir

        self.grid_dir = md.grid_dir

        # TODO: get folders from base model
        self.bel_dir = jp(md.forecasts_dir, study_folder)
        if base_dir is None:
            self.base_dir = jp(os.path.dirname(self.bel_dir), "base", "obj")
        else:
            self.base_dir = base_dir
        self.res_dir = jp(self.bel_dir, "obj")
        self.fig_cca_dir = jp(self.bel_dir, "cca")
        self.fig_pred_dir = jp(self.bel_dir, "uq")

        self.po = PosteriorIO(directory=self.res_dir)

        # Load objects
        f_names = list(
            map(lambda fn: jp(self.res_dir, fn + ".pkl"), ["cca", "d_pca"]))
        self.cca_operator, self.d_pco = list(map(joblib.load, f_names))
        self.h_pco = joblib.load(jp(self.base_dir, "h_pca.pkl"))

        # Inspect transformation between physical and PC space
        dnc0 = self.d_pco.n_pc_cut
        hnc0 = self.h_pco.n_pc_cut

        # Cut desired number of PC components
        d_pc_training, self.d_pc_prediction = self.d_pco.comp_refresh(dnc0)
        self.h_pco.comp_refresh(hnc0)

        # Sampling
        self.n_training = len(d_pc_training)
        self.n_posts = self.base.HyperParameters.n_posts
        self.forecast_posterior = None
        self.h_true_obs = None  # True h in physical space
        self.shape = None
        self.h_pc_true_pred = None  # CCA predicted 'true' h PC
        self.h_pred = None  # 'true' h in physical space

        # 0 contours of posterior WHPA
        self.vertices = None

    # %% Random sample from the posterior
    def sample_posterior(self, n_posts: int = None):
        """
        Extracts n_posts random samples from the posterior.
        :param n_posts: int: Desired number of samples
        :return:
        """

        if n_posts is not None:
            self.n_posts = n_posts

        # Extract n random sample (target pc's).
        # The posterior distribution is computed within the method below.
        self.forecast_posterior = self.po.bel_predict(
            pca_d=self.d_pco,
            pca_h=self.h_pco,
            cca_obj=self.cca_operator,
            n_posts=self.n_posts,
            add_comp=False,
        )

        # Get the true array of the prediction
        # Prediction set - PCA space
        self.shape = self.h_pco.training_shape

    # %% extract 0 contours
    def c0(self, write_vtk: bool = False):
        """
        Extract the 0 contour from the sampled posterior, corresponding to the WHPA delineation
        :param write_vtk: bool: Flag to export VTK files
        """
        *_, x, y = refine_machine(self.x_lim, self.y_lim, self.grf)
        self.vertices = contours_vertices(x, y, self.forecast_posterior)
        if write_vtk:
            vdir = jp(self.fig_pred_dir, "vtk")
            utils.dirmaker(vdir)
            for i, v in enumerate(self.vertices):
                nv = len(v)
                points = vtk.vtkPoints()
                [points.InsertNextPoint(np.insert(c, 2, 0)) for c in v]
                # Create a polydata to store everything in
                poly_data = vtk.vtkPolyData()
                # Add the points to the dataset
                poly_data.SetPoints(points)
                # Create a cell array to store the lines in and add the lines to it
                cells = vtk.vtkCellArray()
                cells.InsertNextCell(nv)
                [cells.InsertCellPoint(k) for k in range(nv)]
                # Add the lines to the dataset
                poly_data.SetLines(cells)
                # Export
                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetInputData(poly_data)

                writer.SetFileName(jp(vdir, f"forecast_posterior_{i}.vtp"))
                writer.Write()
        return x, y

    def objective_function(self):
        """
        Computes the metric between the true WHPA that has been recovered from its n first PCA
        components to allow proper comparison.
        """
        # TODO : Extract this function ?
        # The new idea is to compute the metric with the observed WHPA recovered from it's n first PC.
        n_cut = self.h_pco.n_pc_cut  # Number of components to keep
        # Inverse transform and reshape
        true_image = self.h_pco.custom_inverse_transform(
            self.h_pco.predict_pc, n_cut).reshape(
            (self.shape[1], self.shape[2]))

        method_name = self.base.ED.metric.__name__
        logger.info(f"Quantifying image difference based on {method_name}")
        if method_name == "modified_hausdorff":
            x, y = self.c0()
            to_compare = self.vertices
            true_feature = contours_vertices(x=x, y=y, arrays=true_image)[0]
        else:
            to_compare = self.forecast_posterior
            true_feature = true_image

        # Compute metric between the 'true image' and the n sampled images or images feature
        similarity = np.array(
            [self.base.ED.metric(true_feature, f) for f in to_compare])

        # Save objective_function result
        np.save(jp(self.res_dir, f"{method_name}"), similarity)

        logger.info(f"Similarity : {np.mean(similarity)}")

        return np.mean(similarity)


def measure_info_mode(base: Type[Setup], roots_obs: Root):

    logger.info("Computing ED results")

    wid = list(map(str, [_[0] for _ in base.Wells.combination]))  # Well identifiers (n)
    wm = np.zeros((len(wid), base.HyperParameters.n_posts))

    for r in roots_obs:
        # Starting point = root folder in forecast directory
        droot = os.path.join(base.Directories.forecasts_dir, r)
        for e in wid:  # For each sub folder (well) in the main folder
            # Get the objective function file
            ufp = os.path.join(droot, e, "obj")
            fmhd = os.path.join(ufp, f"{base.ED.metric.__name__}.npy")
            mhd = np.load(fmhd)  # Load MHD
            idw = int(e) - 1  # -1 to respect 0 index (Well index)
            wm[idw] += mhd  # Add MHD at each well

    logger.info("Done")
    np.save(os.path.join(base.Directories.forecasts_dir, f"uq_{base.ED.metric.__name__}.npy"), wm)


def analysis(
        base: Type[Setup],
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

    if roots_obs is None:  # If no observation provided
        if n_training + n_obs <= len(folders):
            # List of m observation roots
            roots_obs = folders[n_training:(n_training + n_obs)]
        else:
            logger.error("Incompatible training/observation numbers")
            return

    for i, r in enumerate(roots_training):
        choices = folders[n_training:].copy()
        if r in roots_obs:
            random_root = np.random.choice(choices)
            roots_training[i] = random_root
            choices.remove(random_root)

    for r in roots_obs:
        if r in roots_training:
            logger.warning(f"obs {r} is located in the training roots")
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
    utils.dirmaker(obj_path)  # Returns bool according to folder status
    if flag_base:
        utils.dirmaker(obj_path, erase=flag_base)
        # Creates main target PCA object
        obj = os.path.join(obj_path, "h_pca.pkl")
        logger.info("Performing base PCA")
        base_pca(
            base=base,
            base_dir=obj_path,
            roots=roots_training,
            test_roots=roots_obs,
            h_pca_obj=obj
        )

    # --------------------------------------------------------

    obs = roots_obs
    base_dir_path = obj_path

    # Resets the target PCA object' predictions to None before starting
    try:
        joblib.load(os.path.join(base_dir_path, "h_pca.pkl")).reset_()
    except FileNotFoundError:
        pass

    combinations = base.Wells.combination.copy()

    global_mean = 0
    total = len(obs)
    for ix, r_ in enumerate(obs):  # For each observation root
        logger.info(f"[{ix+1}/{total}]-{r_}")
        for ixw, c in enumerate(combinations):  # For each wel combination
            logger.info(f"[{ix+1}/{total}]-{r_}-{ixw+1}/{len(combinations)}")
            # PCA decomposition + CCA
            base.Wells.combination = c  # This might not be so optimal
            logger.info("Fit - Transform")
            sf = bel_fit_transform(base=base,
                                   training_roots=roots_training,
                                   test_root=r_)

            # TODO: Extract this
            # Uncertainty analysis
            logger.info("Uncertainty quantification")

            uq = UncertaintyQuantification(
                base=base,
                study_folder=sf,
                base_dir=base_dir_path,
                seed=123456,
            )
            # Sample posterior
            logger.info("Sample posterior")
            uq.sample_posterior(n_posts=base.HyperParameters.n_posts)
            logger.info("Similarity measure")
            mean = uq.objective_function()
            global_mean += mean

        # Resets the target PCA object' predictions to None before moving on to the next root
        joblib.load(os.path.join(base_dir_path, "h_pca.pkl")).reset_()

    return global_mean
