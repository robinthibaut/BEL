#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
import shutil
from os.path import join as jp
from typing import List, Type

import joblib
import numpy as np
import vtk
from sklearn.neighbors import KernelDensity
from loguru import logger

from .. import utils
from ..config import Setup
from ..utils import Root, Function, Combination
from ..goggles import mode_histo
from ..learning.bel_pipeline import bel_fit_transform, base_pca, PosteriorIO
from ..spatial import (
    binary_polygon,
    contours_vertices,
    grid_parameters,
    refine_machine,
)

__all__ = ['UncertaintyQuantification', 'by_mode', 'scan_roots', 'analysis']


class UncertaintyQuantification:
    def __init__(
            self,
            base: Type[Setup],
            study_folder: str,
            base_dir: str = None,
            wel_comb: list = None,
            metric: Function = None,
            seed: int = None,
    ):
        """

        :param base: class: Base object (inventory)
        :param study_folder: str: Name of the root uuid in the 'forecast' directory on which UQ will be performed
        :param base_dir: str: Path to base directory
        :param wel_comb: list: List of data source combinations
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

        self.wel_comb = wel_comb

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

        # Metric
        self.metric = metric

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

    # %% Kernel density
    def kernel_density(self):
        # Scatter plot vertices
        # nn = sample_n
        # plt.plot(vertices[nn][:, 0], vertices[nn][:, 1], 'o-')
        # plt.show()

        # Grid geometry
        xmin = self.x_lim[0]
        xmax = self.x_lim[1]
        ymin = self.y_lim[0]
        ymax = self.y_lim[1]
        # Create a structured grid to estimate kernel density
        # TODO: create a function to copy/paste values on differently refined grids
        # Prepare the Plot instance with right dimensions
        grf_kd = 4
        cell_dim = grf_kd
        xgrid = np.arange(xmin, xmax, cell_dim)
        ygrid = np.arange(ymin, ymax, cell_dim)
        X, Y = np.meshgrid(xgrid, ygrid)
        # x, y coordinates of the grid cells vertices
        xy = np.vstack([X.ravel(), Y.ravel()]).T

        # Define a disk within which the KDE will be performed to save time
        # TODO: Move this to parameter file
        x0, y0, radius = 1000, 500, 200
        r = np.sqrt((xy[:, 0] - x0) ** 2 + (xy[:, 1] - y0) ** 2)
        inside = r < radius
        xyu = xy[inside]  # Create mask

        # Perform KDE
        bw = 1.0  # Arbitrary 'smoothing' parameter
        # Reshape coordinates
        x_stack = np.hstack([vi[:, 0] for vi in self.vertices])
        y_stack = np.hstack([vi[:, 1] for vi in self.vertices])
        # Final array np.array([[x0, y0],...[xn,yn]])
        xykde = np.vstack([x_stack, y_stack]).T
        kde = KernelDensity(kernel="gaussian",
                            bandwidth=bw).fit(  # Fit kernel density
            xykde)
        # Sample at the desired grid cells
        score = np.exp(kde.score_samples(xyu))

        def score_norm(sc, max_score=None):
            """
            Normalizes the KDE scores.
            """
            sc -= sc.min()
            sc /= sc.max()

            sc += 1
            sc = sc ** -1

            sc -= sc.min()
            sc /= sc.max()

            return sc

        # Normalize
        score = score_norm(score)

        # Assign the computed scores to the grid
        z = np.full(inside.shape, 1, dtype=float)  # Create array filled with 1
        z[inside] = score
        # Flip to correspond to actual distribution.
        z = np.flipud(z.reshape(X.shape))

        return z

    # %% New approach : stack binary WHPA
    def uq_binary_stack(self):
        """
        Takes WHPA vertices and binarizes the image (e.g. 1 inside, 0 outside WHPA).
        """
        xys, nrow, ncol = grid_parameters(x_lim=self.x_lim,
                                          y_lim=self.y_lim,
                                          grf=self.grf)  # Initiate SD object
        # Create binary images of WHPA stored in bin_whpa
        bin_whpa = [
            binary_polygon(xys,
                           nrow,
                           ncol,
                           pzs=p,
                           inside=1 / self.n_posts,
                           outside=0) for p in self.vertices
        ]
        big_sum = np.sum(bin_whpa, axis=0)  # Stack them
        b_low = np.where(big_sum == 0, 1, big_sum)  # Replace 0 values by 1
        b_low = np.flipud(b_low)

        # Save result
        np.save(jp(self.res_dir, "bin"), b_low)

    def objective_function(self):
        """
        Computes the metric between the true WHPA that has been recovered from its n first PCA
        components to allow proper comparison.
        """

        # The new idea is to compute the metric with the observed WHPA recovered from it's n first PC.
        n_cut = self.h_pco.n_pc_cut  # Number of components to keep
        # Inverse transform and reshape
        true_image = self.h_pco.custom_inverse_transform(
            self.h_pco.predict_pc, n_cut).reshape(
            (self.shape[1], self.shape[2]))

        method_name = self.metric.__name__
        if method_name == "modified_hausdorff":
            x, y = self.c0()
            to_compare = self.vertices
            true_feature = contours_vertices(x=x, y=y, arrays=true_image)[0]
        else:
            to_compare = self.forecast_posterior
            true_feature = true_image

        # Compute metric between the 'true image' and the n sampled images or images feature
        similarity = np.array(
            [self.metric(true_feature, f) for f in to_compare])

        # Save objective_function result
        np.save(jp(self.res_dir, f"{method_name}"), similarity)

        return np.mean(similarity)


def by_mode(root: Root):
    """
    Computes the combined amount of information for n observations.
    see also
    https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
    :param root: list: List containing the roots whose wells contributions will be taken into account.
    :return:
    """

    if not isinstance(root, (list, tuple)):
        root: list = [root]

    # Deals with the fact that only one root might be selected
    fig_name = "average"
    an_i = 0  # Annotation index
    if len(root) == 1:
        fig_name = root[0]
        an_i = 2

    wid = list(map(str, Setup.Wells.combination))  # Well identifiers (n)
    # Summed MHD when well #i appears
    wm = np.zeros((len(wid), Setup.HyperParameters.n_posts))
    colors = Setup.Wells.colors

    for r in root:  # For each root
        # Starting point = root folder in forecast directory
        droot = os.path.join(Setup.Directories.forecasts_dir, r)
        for e in wid:  # For each subfolder (well) in the main folder
            # Get the MHD file
            fmhd = os.path.join(droot, e, "obj", "haus.npy")
            mhd = np.load(fmhd)  # Load MHD
            idw = int(e) - 1  # -1 to respect 0 index (Well index)
            wm[idw] += mhd  # Add MHD at each well

    mode_histo(colors=colors, an_i=an_i, wm=wm, fig_name=fig_name)


def scan_roots(base: Type[Setup],
               training: Root,
               obs: Root,
               combinations: List[int],
               metric: Function = None,
               base_dir_path: str = None) -> float:
    """
    Scan forward roots and perform base decomposition.
    :param base: class: Base class (inventory)
    :param training: list: List of uuid of each root for training
    :param obs: list: List of uuid of each root for observation
    :param combinations: list: List of wells combinations, e.g. [[1, 2, 3, 4, 5, 6]]
    :param metric: Function: Metric method
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
            sf = bel_fit_transform(base=base,
                                   training_roots=training,
                                   test_root=r_,
                                   well_comb=c)
            # Uncertainty analysis
            uq = UncertaintyQuantification(
                base=base,
                study_folder=sf,
                base_dir=base_dir_path,
                wel_comb=c,
                metric=metric,
                seed=123456,
            )
            # Sample posterior
            uq.sample_posterior(n_posts=base.HyperParameters.n_posts)
            # uq.c0(write_vtk=False)  # Extract 0 contours
            uq.metric = metric
            mean = uq.objective_function()
            global_mean += mean

        # Resets the target PCA object' predictions to None before moving on to the next root
        joblib.load(os.path.join(base_dir_path, "h_pca.pkl")).reset_()

    return global_mean


def analysis(
        base: Type[Setup],
        comb: Combination = None,
        n_training: int = 200,
        n_obs: int = 50,
        metric: Function = None,
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
    :param metric: Function: Metric method
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
    fb = utils.dirmaker(
        obj_path)  # Returns bool according to folder status
    if flag_base:
        utils.dirmaker(obj_path, erase=flag_base)
        # Creates main target PCA object
        obj = os.path.join(obj_path, "h_pca.pkl")
        base_pca(
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
        metric=metric,
        base_dir_path=obj_path,
    )

    if wipe:
        shutil.rmtree(Setup.Directories.forecasts_dir)

    return roots_training, roots_obs, global_mean
