#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
import shutil
from os.path import join as jp
from typing import Type

import joblib
import numpy as np
import vtk
from loguru import logger

from .. import utils
from ..config import Setup
from ..utils import Root
from ..learning.bel_pipeline import bel_fit_transform, base_pca, PosteriorIO
from ..spatial import (
    contours_vertices,
    refine_machine,
)

__all__ = [
    "UncertaintyQuantification",
    "measure_info_mode",
    "objective_function",
    "compute_metric",
]


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
            self.base_dir = jp(md.forecasts_dir, "base")
        else:
            self.base_dir = base_dir
        self.res_dir = jp(self.bel_dir, "obj")
        self.fig_cca_dir = jp(self.bel_dir, "cca")
        self.fig_pred_dir = jp(self.bel_dir, "uq")

        self.po = PosteriorIO(directory=self.res_dir)

        # Load objects
        f_names = list(map(lambda fn: jp(self.res_dir, f"{fn}.pkl"), ["cca", "d_pca"]))
        self.cca_operator, self.d_pco = list(map(joblib.load, f_names))

        try:
            self.h_pco = joblib.load(jp(self.base_dir, "h_pca.pkl"))
            logger.info("Base target training object reloaded")
        except FileNotFoundError:
            self.h_pco = None
            logger.info("Base target training object not found")

        # Inspect transformation between physical and PC space
        dnc0 = self.d_pco.n_pc_cut

        # Cut desired number of PC components
        d_pc_training, self.d_pc_prediction = self.d_pco.comp_refresh(dnc0)

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

    def analysis(
            self,
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
        II. Define one or more 'observation' root(s) (roots_obs in params).
        III. Perform PCA decomposition on the training targets and store the output in the 'base' folder,
        to avoid recomputing it every time.
        IV. Given n combinations of data source, apply BEL approach n times and perform uncertainty quantification.

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
        md = self.base.Directories.hydro_res_dir
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

        if roots_obs is None:  # If no observation provided
            if n_training + n_obs <= len(folders):
                # List of m observation roots
                roots_obs = folders[n_training: (n_training + n_obs)]
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
        # /!\ Danger zone /!\
        if wipe:
            try:
                shutil.rmtree(self.base.Directories.forecasts_dir)
            except FileNotFoundError:
                pass

        if flag_base:
            utils.dirmaker(self.base_dir, erase=flag_base)
            # Creates main target PCA object
            obj = os.path.join(self.base_dir, "h_pca.pkl")
            logger.info("Performing base PCA")
            self.h_pco = base_pca(
                base=self.base,
                base_dir=self.base_dir,
                training_roots=roots_training,
                test_roots=roots_obs,
                h_pca_obj_path=obj,
            )

        # Fit transform
        combinations = self.base.Wells.combination.copy()
        total = len(roots_obs)
        for ix, r_ in enumerate(roots_obs):  # For each observation root
            logger.info(f"[{ix + 1}/{total}]-{r_}")
            for ixw, c in enumerate(combinations):  # For each wel combination
                logger.info(f"[{ix + 1}/{total}]-{r_}-{ixw + 1}/{len(combinations)}")
                # PCA decomposition + CCA
                self.base.Wells.combination = c  # This might not be so optimal
                logger.info("Fit - Transform")
                bel_fit_transform(base=self.base, training_roots=roots_training, test_root=r_)

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
        self.shape = self.h_pco.training_df.attrs["physical_shape"]

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


def objective_function(uq: UncertaintyQuantification, metric):
    """
    Computes the metric between the true WHPA that has been recovered from its n first PCA
    components to allow proper comparison.
    """
    # The idea is to compute the metric with the observed WHPA recovered from it's n first PC.
    n_cut = uq.h_pco.n_pc_cut  # Number of components to keep
    # Inverse transform and reshape
    # FIXME: Problem is that uq.h_pco_predict_pc is None
    true_image = uq.h_pco.custom_inverse_transform(uq.h_pco.predict_pc, n_cut).reshape(
        (uq.shape[1], uq.shape[2])
    )

    method_name = metric.__name__
    logger.info(f"Quantifying image difference based on {method_name}")
    if method_name == "modified_hausdorff":
        x, y = uq.c0()
        to_compare = uq.vertices
        true_feature = contours_vertices(x=x, y=y, arrays=true_image)[0]
    elif method_name == "structural_similarity":
        to_compare = uq.forecast_posterior
        true_feature = true_image
    else:
        logger.error("Metric name not recognized.")
        to_compare = None
        true_feature = None

    # Compute metric between the 'true image' and the n sampled images or images feature
    similarity = np.array([metric(true_feature, f) for f in to_compare])

    # Save objective_function result
    np.save(jp(uq.res_dir, f"{method_name}"), similarity)

    logger.info(f"Similarity : {np.mean(similarity)}")

    return np.mean(similarity)


def measure_info_mode(base: Type[Setup], roots_obs: Root, metric):
    """Scan the computed metric files and process them based on the mode"""
    logger.info("Computing ED results")

    wid = list(map(str, [_[0] for _ in base.Wells.combination]))  # Well identifiers (n)
    wm = np.zeros((len(wid), base.HyperParameters.n_posts))

    for r in roots_obs:
        # Starting point = root folder in forecast directory
        droot = os.path.join(base.Directories.forecasts_dir, r)
        for e in wid:  # For each sub folder (well) in the main folder
            # Get the objective function file
            ufp = os.path.join(droot, e, "obj")
            fmhd = os.path.join(ufp, f"{metric.__name__}.npy")
            mhd = np.load(fmhd)  # Load MHD
            idw = int(e) - 1  # -1 to respect 0 index (Well index)
            wm[idw] += mhd  # Add MHD at each well

    logger.info("Done")
    np.save(
        os.path.join(base.Directories.forecasts_dir, f"uq_{metric.__name__}.npy"), wm
    )


def compute_metric(
    base: Type[Setup], roots_obs: Root, combinations: list, metric, base_dir: str = None
):
    if base_dir is None:
        base_dir = os.path.join(base.Directories.forecasts_dir, "base")

    global_mean = 0
    total = len(roots_obs)
    for ix, r_ in enumerate(roots_obs):  # For each observation root
        logger.info(f"[{ix + 1}/{total}]-{r_}")
        for ixw, c in enumerate(combinations):  # For each wel combination
            logger.info(f"[{ix + 1}/{total}]-{r_}-{ixw + 1}/{len(combinations)}")
            # Uncertainty analysis
            logger.info("Uncertainty quantification")
            study_folder = os.path.join(
                base.Directories.forecasts_dir, f"{r_}", "".join(list(map(str, c)))
            )
            uq = UncertaintyQuantification(
                base=base,
                base_dir=base_dir,
                study_folder=study_folder,
                seed=123456,
            )
            # Sample posterior
            logger.info("Sample posterior")
            uq.sample_posterior(n_posts=base.HyperParameters.n_posts)
            logger.info("Similarity measure")
            mean = objective_function(uq, metric)
            global_mean += mean

    return global_mean
