#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
from os.path import join as jp
from typing import Type

import joblib
import numpy as np
import pandas as pd
import vtk
from loguru import logger
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.pipeline import Pipeline

from .. import utils
from ..config import Setup
from ..utils import Root
from ..learning.bel import BEL
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
        seed: int = None,
    ):
        """
        :param base: class: Base object (inventory)
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

        # Number of CCA components is chosen as the min number of PC
        n_pc_pred, n_pc_targ = (
            base.HyperParameters.n_pc_predictor,
            base.HyperParameters.n_pc_target,
        )

        # Pipeline before CCA
        self.X_pre_processing = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("pca", PCA()),
            ]
        )
        self.Y_pre_processing = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("pca", PCA()),
            ]
        )

        self.cca = CCA(
            n_components=min(n_pc_targ, n_pc_pred), max_iter=500 * 20, tol=1e-6
        )

        self.X_post_processing = Pipeline(
            [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
        )
        self.Y_post_processing = Pipeline(
            [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
        )

        self.bel = BEL(
            X_pre_processing=self.X_pre_processing,
            X_post_processing=self.X_post_processing,
            Y_pre_processing=self.Y_pre_processing,
            Y_post_processing=self.Y_post_processing,
            cca=self.cca,
        )

        self.X_obs = None

        # Sampling
        self.n_posts = self.base.HyperParameters.n_posts
        self.forecast_posterior = None
        self.Y_true_obs = None  # True h in physical space
        self.shape = None
        self.Y_pc_true_pred = None  # CCA predicted 'true' h PC
        self.Y_pred = None  # 'true' h in physical space

        # 0 contours of posterior WHPA
        self.vertices = None

    def analysis(
        self,
        roots_training: Root = None,
        roots_obs: Root = None,
    ):
        """
        I. First, defines the roots for training from simulations in the hydro results directory.
        II. Define one or more 'observation' root(s) (roots_obs in params).
        III. Perform PCA decomposition on the training targets and store the output in the 'base' folder,
        to avoid recomputing it every time.
        IV. Given n combinations of data source, apply BEL approach n times and perform uncertainty quantification.

        :param roots_training: list: List of roots considered as training.
        :param roots_obs: list: List of roots considered as observations.
        :return: list: List of training roots, list: List of observation roots

        """

        # Directories
        md = self.base.Directories
        combinations = [self.base.Wells.combination.copy()]
        total = len(roots_obs)
        for ix, test_root in enumerate(roots_obs):  # For each observation root
            logger.info(f"[{ix + 1}/{total}]-{test_root}")
            # Directory in which to load forecasts
            bel_dir = jp(md.forecasts_dir, test_root)

            for ixw, c in enumerate(combinations):  # For each wel combination
                logger.info(
                    f"[{ix + 1}/{total}]-{test_root}-{ixw + 1}/{len(combinations)}"
                )

                new_dir = "".join(
                    list(map(str, self.base.Wells.combination))
                )  # sub-directory for forecasts
                sub_dir = jp(bel_dir, new_dir)

                # %% Folders
                obj_dir = jp(sub_dir, "obj")
                fig_data_dir = jp(sub_dir, "data")
                fig_pca_dir = jp(sub_dir, "pca")
                fig_cca_dir = jp(sub_dir, "cca")
                fig_pred_dir = jp(sub_dir, "uq")

                # %% Creates directories
                [
                    utils.dirmaker(f)
                    for f in [
                        obj_dir,
                        fig_data_dir,
                        fig_pca_dir,
                        fig_cca_dir,
                        fig_pred_dir,
                    ]
                ]

                # Load training dataset
                # %% Select wells:
                selection = [wc - 1 for wc in self.base.Wells.combination]
                tc_training = tc_training[:, selection, :]
                tc_test = tc_test[:, selection, :]
                # Convert to dataframes
                training_df_predictor = utils.i_am_framed(
                    array=tc_training, ids=roots_training
                )

                # %% Fit fit_transform
                # PCA decomposition + CCA
                self.base.Wells.combination = c  # This might not be so optimal
                self.bel.fit(X=training_df_predictor, Y=h_pco.training_df)
                joblib.dump(
                    self.bel.cca, jp(obj_dir, "bel.pkl")
                )  # Save the fitted CCA operator
                msg = f"model trained and saved in {obj_dir}"
                logger.info(msg)

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
        Y_posts_gaussian = self.bel.predict(self.X_obs)

        self.forecast_posterior = self.bel.inverse_transform(
            Y_pred=Y_posts_gaussian.reshape(1, -1),
        )
        # Get the true array of the prediction
        # Prediction set - PCA space
        self.shape = self.h_pco.training_df.attrs["physical_shape"]

        return self.bel.posterior_mean, self.bel.posterior_covariance

    # %% extract 0 contours
    def contour_extract(self, write_vtk: bool = False):
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
    # Inverse fit_transform and reshape
    # FIXME: Problem is that uq.h_pco_predict_pc is None
    true_image = uq.h_pco.custom_inverse_transform(uq.h_pco.predict_pc, n_cut).reshape(
        (uq.shape[1], uq.shape[2])
    )

    method_name = metric.__name__
    logger.info(f"Quantifying image difference based on {method_name}")
    if method_name == "modified_hausdorff":
        x, y = uq.contour_extract()
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
        base_dir = base.Directories.forecasts_base_dir

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
