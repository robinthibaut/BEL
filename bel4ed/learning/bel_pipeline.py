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
import warnings
from os.path import join as jp

import joblib
import numpy as np
from sklearn.preprocessing import PowerTransformer

from .. import utils
from ..config import Setup, Root, Combination

from ..processing import curve_interpolation

from ..algorithms import CCA, signed_distance
from ..spatial import grid_parameters, get_block
from ..processing import PC


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
        tc0, _, _ = utils.data_loader(roots=roots, d=True)
        # tc0 = breakthrough curves with shape (n_sim, n_wells, n_time_steps)
        # pzs = WHPA
        # roots_ = simulation id
        # Subdivide d in an arbitrary number of time steps:
        # tc has shape (n_sim, n_wells, n_time_steps)
        tc = curve_interpolation(tc0=tc0)
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
        _, pzs, r = utils.data_loader(roots=roots, h=True)
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
    It saves results as pkl objects that have to be loaded in the _forecast_error.py script to perform predictions.

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
        utils.dirmaker(f) for f in
        [obj_dir, fig_data_dir, fig_pca_dir, fig_cca_dir, fig_pred_dir]
    ]

    # Load training data
    # Refined breakthrough curves data file
    tsub = jp(base_dir, "training_curves.npy")
    if not os.path.exists(tsub):
        # Loads the results:
        tc0, _, _ = utils.data_loader(res_dir=res_dir,
                                      roots=training_roots,
                                      d=True)
        # tc0 = breakthrough curves with shape (n_sim, n_wells, n_time_steps)
        # pzs = WHPA's
        # roots_ = simulations id's
        # Subdivide d in an arbitrary number of time steps:
        # tc has shape (n_sim, n_wells, n_time_steps)
        tc = curve_interpolation(tc0=tc0, n_time_steps=200)
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
    tc0, _, _ = utils.data_loader(res_dir=res_dir,
                                  test_roots=test_root,
                                  d=True)
    # Subdivide d in an arbitrary number of time steps:
    tcp = curve_interpolation(tc0=tc0, n_time_steps=n_time_steps)
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
    _, pzs, _ = utils.data_loader(roots=test_root, h=True)
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
        utils.dirmaker(fig_dir)
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


class PosteriorIO:
    """
    Heart of the framework.
    """

    def __init__(self, directory: str = None):
        self.posterior_mean = None
        self.posterior_covariance = None
        self.seed = None
        self.n_posts = None
        self.normalize_h = PowerTransformer(method="yeo-johnson",
                                            standardize=True)
        self.normalize_d = PowerTransformer(method="yeo-johnson",
                                            standardize=True)
        self.directory = directory

    def mvn_inference(
            self,
            h_cca_training_gaussian,
            d_cca_training,
            d_pc_training,
            d_rotations,
            d_cca_prediction,
    ):
        """
        Estimating posterior mean and covariance of the target.
        .. [1] A. Tarantola. Inverse Problem Theory and Methods for Model Parameter Estimation.
               SIAM, 2005. Pages: 70-71
        :param h_cca_training_gaussian: Canonical Variate of the training target, gaussian-distributed
        :param d_cca_training: Canonical Variate of the training data
        :param d_pc_training: Principal Components of the training data
        :param d_rotations: CCA rotations of the training data (project original data to canonical space)
        :param d_cca_prediction: Canonical Variate of the observation
        :return: h_posterior_mean, h_posterior_covariance
        :raise ValueError: An exception is thrown if the shape of input arrays are not consistent.
        """

        # TODO: add dimension check
        if isinstance(h_cca_training_gaussian, (list, tuple, np.ndarray)):
            # Shape = (n_components_CCA, n_training)
            shctg = np.shape(h_cca_training_gaussian)
        if isinstance(d_cca_training, (list, tuple, np.ndarray)):
            # Shape = (n_components_CCA, n_training)
            sdct = np.shape(d_cca_training)
        if isinstance(d_pc_training, (list, tuple, np.ndarray)):
            # Shape = (n_training, n_components_PCA)
            sdpt = np.shape(d_pc_training)
        if isinstance(d_rotations, (list, tuple, np.ndarray)):
            # Shape = (n_components_PCA_d, n_components_CCA_h)
            sdr = np.shape(d_rotations)
        if isinstance(d_cca_prediction, (list, tuple, np.ndarray)):
            sdcp = np.shape(d_cca_prediction)  # Shape = (n_components_CCA, 1)

        # Size of the set
        n_training = d_cca_training.shape[0]

        # Computation of the posterior mean in Canonical space
        h_mean = np.mean(h_cca_training_gaussian, axis=0)  # (n_comp_CCA, 1)
        # Mean is 0, as expected.
        h_mean = np.where(np.abs(h_mean) < 1e-12, 0, h_mean)

        # Evaluate the covariance in h (in Canonical space)
        # Very close to the Identity matrix
        # (n_comp_CCA, n_comp_CCA)
        h_cov_operator = np.cov(h_cca_training_gaussian.T)

        # Evaluate the covariance in d (here we assume no data error, so C is identity times a given factor)
        # Number of PCA components for the curves
        x_dim = np.size(d_pc_training, axis=1)
        noise = 0.01
        # I matrix. (n_comp_PCA, n_comp_PCA)
        d_cov_operator = np.eye(x_dim) * noise
        # (n_comp_CCA, n_comp_CCA)
        d_noise_covariance = d_rotations.T @ d_cov_operator @ d_rotations

        # Linear modeling h to d (in canonical space) with least-square criterion.
        # Pay attention to the transpose operator.
        # Computes the vector g that approximately solves the equation h @ g = d.
        g = np.linalg.lstsq(h_cca_training_gaussian,
                            d_cca_training,
                            rcond=None)[0].T
        # Replace values below threshold by 0.
        g = np.where(np.abs(g) < 1e-12, 0, g)  # (n_comp_CCA, n_comp_CCA)

        # Modeling error due to deviations from theory
        # (n_components_CCA, n_training)
        d_ls_predicted = h_cca_training_gaussian @ g.T
        d_modeling_mean_error = np.mean(d_cca_training - d_ls_predicted,
                                        axis=0)  # (n_comp_CCA, 1)
        d_modeling_error = (d_cca_training - d_ls_predicted -
                            np.tile(d_modeling_mean_error, (n_training, 1)))
        # (n_comp_CCA, n_training)

        # Information about the covariance of the posterior distribution in Canonical space.
        d_modeling_covariance = np.cov(
            d_modeling_error.T)  # (n_comp_CCA, n_comp_CCA)

        # Build block matrix
        s11 = h_cov_operator
        s12 = h_cov_operator @ g.T
        s21 = g @ h_cov_operator
        s22 = g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance
        block = np.block([[s11, s12], [s21, s22]])

        # Inverse
        delta = np.linalg.pinv(block)
        # Partition block
        d11 = get_block(delta, 1)
        d12 = get_block(delta, 2)

        # Observe that posterior covariance does not depend on observed d.
        h_posterior_covariance = np.linalg.pinv(d11)
        # Computing the posterior mean is simply a linear operation, given precomputed posterior covariance.
        h_posterior_mean = h_posterior_covariance @ (
                d11 @ h_mean -
                d12 @ (d_cca_prediction[0] - d_modeling_mean_error - h_mean @ g.T))

        # test = np.block([[d11, d12], [d21, d22]])
        # plt.matshow(test, cmap='coolwarm')
        # plt.colorbar()
        # plt.show()

        # Also works:
        # Inverse of the sample covariance matrix of d ( Sig dd )
        # ddd_inv = np.linalg.pinv(g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance)
        # h_posterior_covariance = h_cov_operator - \
        #     h_cov_operator @ g.T @ ddd_inv @ g @ h_cov_operator
        #
        # h_posterior_mean = \
        #     h_mean + h_cov_operator @ g.T @ ddd_inv @ (d_cca_prediction[0] - d_modeling_mean_error - h_mean @ g.T)

        # h_posterior_covariance = (h_posterior_covariance + h_posterior_covariance.T) / 2  # (n_comp_CCA, n_comp_CCA)

        self.posterior_mean = h_posterior_mean  # (n_comp_CCA,)
        # (n_comp_CCA, n_comp_CCA)
        self.posterior_covariance = h_posterior_covariance

    def back_transform(
            self,
            h_posts_gaussian,
            cca_obj,
            pca_h,
            n_posts: int,
            add_comp: bool = False,
            save_target_pc: bool = False,
    ):
        """
        Back-transforms the sampled gaussian distributed posterior h to their physical space.
        :param h_posts_gaussian:
        :param cca_obj:
        :param pca_h:
        :param n_posts:
        :param add_comp:
        :param save_target_pc:
        :return: forecast_posterior
        """
        # This h_posts gaussian need to be inverse-transformed to the original distribution.
        # We get the CCA scores.

        # h_posts = self.processing.gaussian_inverse(h_posts_gaussian)  # (n_components, n_samples)
        h_posts = self.normalize_h.inverse_transform(
            h_posts_gaussian)  # (n_components, n_samples)

        # Calculate the values of hf, i.e. reverse the canonical correlation, it always works if dimf > dimh
        # The value of h_pca_reverse are the score of PCA in the forecast space.
        # To reverse data in the original space, perform the matrix multiplication between the data in the CCA space
        # with the y_loadings matrix. Because CCA scales the input, we must multiply the output by the y_std dev
        # and add the y_mean.
        # FIXME: Deprecation warning here about y_std_ and y_mean_
        h_pca_reverse = (
                np.matmul(h_posts, cca_obj.y_loadings_.T) * cca_obj.y_std_ +
                cca_obj.y_mean_)

        # Whether to add or not the rest of PC components
        if add_comp:  # TODO: double check
            rnpc = np.array([pca_h.random_pc(n_posts) for _ in range(n_posts)
                             ])  # Get the extra components
            h_pca_reverse = np.array([
                np.concatenate((h_pca_reverse[i], rnpc[i]))
                for i in range(n_posts)
            ])  # Insert it

        if save_target_pc:
            fname = jp(self.directory, "target_pc.npy")
            np.save(fname, h_pca_reverse)

        # Generate forecast in the initial dimension and reshape.
        forecast_posterior = pca_h.custom_inverse_transform(
            h_pca_reverse).reshape(
            (n_posts, pca_h.training_shape[1], pca_h.training_shape[2]))

        return forecast_posterior

    def random_sample(self, n_posts: int = None):
        """

        :param n_posts:
        :return:
        """
        if n_posts is None:
            n_posts = self.n_posts
        # Draw n_posts random samples from the multivariate normal distribution :
        # Pay attention to the transpose operator
        np.random.seed(self.seed)
        h_posts_gaussian = np.random.multivariate_normal(
            mean=self.posterior_mean,
            cov=self.posterior_covariance,
            size=n_posts)
        return h_posts_gaussian

    def bel_predict(self,
                    pca_d,
                    pca_h,
                    cca_obj,
                    n_posts: int,
                    add_comp: bool = False):
        """
        Make predictions, in the BEL fashion.
        :param pca_d: PCA object for observations.
        :param pca_h: PCA object for targets.
        :param cca_obj: CCA object.
        :param n_posts: Number of posteriors to extract.
        :param add_comp: Flag to add remaining components.
        :return: forecast_posterior
        """

        if self.posterior_mean is None and self.posterior_covariance is None:
            # Cut desired number of PC components
            d_pc_training, d_pc_prediction = pca_d.comp_refresh(pca_d.n_pc_cut)
            h_pc_training, _ = pca_h.comp_refresh(pca_h.n_pc_cut)

            # observation data for prediction sample
            d_pc_obs = d_pc_prediction[0]

            # Transform to canonical space
            d_cca_training, h_cca_training = cca_obj.transform(
                d_pc_training, h_pc_training)
            # d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T

            # Ensure Gaussian distribution in d_cca_training
            d_cca_training = self.normalize_d.fit_transform(d_cca_training)

            # Ensure Gaussian distribution in h_cca_training
            # h_cca_training_gaussian = self.processing.gaussian_distribution(h_cca_training)
            h_cca_training = self.normalize_h.fit_transform(h_cca_training)

            # Get the rotation matrices
            d_rotations = cca_obj.x_rotations_

            # Project observed data into canonical space.
            d_cca_prediction = cca_obj.transform(d_pc_obs.reshape(1, -1))

            d_cca_prediction = self.normalize_d.transform(d_cca_prediction)

            # Estimate the posterior mean and covariance (Tarantola)

            self.mvn_inference(
                h_cca_training,
                d_cca_training,
                d_pc_training,
                d_rotations,
                d_cca_prediction,
            )

            # Set the seed for later use
            if self.seed is None:
                self.seed = np.random.randint(2 ** 32 - 1, dtype="uint32")

            if n_posts is None:
                self.n_posts = Setup.HyperParameters.n_posts
            else:
                self.n_posts = n_posts

            # Saves this postio object to avoid saving large amounts of 'forecast_posterior'
            # This allows to reload this object later on and resample using the same seed.
            joblib.dump(self, jp(self.directory, "post.pkl"))

        # Sample the inferred multivariate gaussian distribution
        h_posts_gaussian = self.random_sample(self.n_posts)

        # Back-transform h posterior to the physical space
        forecast_posterior = self.back_transform(
            h_posts_gaussian=h_posts_gaussian,
            cca_obj=cca_obj,
            pca_h=pca_h,
            n_posts=self.n_posts,
            add_comp=add_comp,
        )

        return forecast_posterior
