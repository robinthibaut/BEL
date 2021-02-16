#  Copyright (c) 2021. Robin Thibaut, Ghent University
import math
import os
import warnings
from os.path import join as jp

import joblib
import numpy as np
import pandas as pd
from pysgems.algo.sgalgo import XML
from pysgems.dis.sgdis import Discretize
from pysgems.io.sgio import PointSet
from pysgems.sgems import sg
from scipy import stats, ndimage, integrate
from sklearn.preprocessing import PowerTransformer

from experiment._core import setup
from experiment.spatial.grid import get_block
from experiment.toolbox.filesio import data_read


class KDE:
    """
    Bivariate kernel density estimator.
    This class is adapted from the class of the same name in the package Seaborn 0.11.1
    https://seaborn.pydata.org/generated/seaborn.kdeplot.html
       """

    def __init__(
            self, *,
            bw_method=None,
            bw_adjust=1,
            gridsize=200,
            cut=3,
            clip=None,
            cumulative=False,
    ):
        """Initialize the estimator with its parameters.

        Parameters
        ----------
        bw_method : string, scalar, or callable, optional
            Method for determining the smoothing bandwidth to use; passed to
            :class:`scipy.stats.gaussian_kde`.
        bw_adjust : number, optional
            Factor that multiplicatively scales the value chosen using
            ``bw_method``. Increasing will make the curve smoother. See Notes.
        gridsize : int, optional
            Number of points on each dimension of the evaluation grid.
        cut : number, optional
            Factor, multiplied by the smoothing bandwidth, that determines how
            far the evaluation grid extends past the extreme datapoints. When
            set to 0, truncate the curve at the data limits.
        clip : pair of numbers None, or a pair of such pairs
            Do not evaluate the density outside of these limits.
        cumulative : bool, optional
            If True, estimate a cumulative distribution function.

        """
        if clip is None:
            clip = None, None

        self.bw_method = bw_method
        self.bw_adjust = bw_adjust
        self.gridsize = gridsize
        self.cut = cut
        self.clip = clip
        self.cumulative = cumulative

        self.support = None

    @staticmethod
    def _define_support_grid(x, bw, cut, clip, gridsize):
        """Create the grid of evaluation points depending for vector x."""
        clip_lo = -np.inf if clip[0] is None else clip[0]
        clip_hi = +np.inf if clip[1] is None else clip[1]
        gridmin = max(x.min() - bw * cut, clip_lo)
        gridmax = min(x.max() + bw * cut, clip_hi)
        return np.linspace(gridmin, gridmax, gridsize)

    def _define_support_univariate(self, x, weights):
        """Create a 1D grid of evaluation points."""
        kde = self._fit(x, weights)
        bw = np.sqrt(kde.covariance.squeeze())
        grid = self._define_support_grid(
            x, bw, self.cut, self.clip, self.gridsize
        )
        return grid

    def _define_support_bivariate(self, x1, x2, weights):
        """Create a 2D grid of evaluation points."""
        clip = self.clip
        if clip[0] is None or np.isscalar(clip[0]):
            clip = (clip, clip)

        kde = self._fit([x1, x2], weights)
        bw = np.sqrt(np.diag(kde.covariance).squeeze())

        grid1 = self._define_support_grid(
            x1, bw[0], self.cut, clip[0], self.gridsize
        )
        grid2 = self._define_support_grid(
            x2, bw[1], self.cut, clip[1], self.gridsize
        )

        return grid1, grid2

    def define_support(self, x1, x2=None, weights=None, cache=True):
        """Create the evaluation grid for a given data set."""
        if x2 is None:
            support = self._define_support_univariate(x1, weights)
        else:
            support = self._define_support_bivariate(x1, x2, weights)

        if cache:
            self.support = support

        return support

    def _fit(self, fit_data, weights=None):
        """Fit the scipy kde"""
        fit_kws = {"bw_method": self.bw_method}
        if weights is not None:
            fit_kws["weights"] = weights

        kde = stats.gaussian_kde(fit_data, **fit_kws)
        kde.set_bandwidth(kde.factor * self.bw_adjust)

        return kde

    def _eval_univariate(self, x, weights=None):
        """Fit and evaluate on univariate data."""
        support = self.support
        if support is None:
            support = self.define_support(x, cache=True)

        kde = self._fit(x, weights)

        if self.cumulative:
            s_0 = support[0]
            density = np.array([
                kde.integrate_box_1d(s_0, s_i) for s_i in support
            ])
        else:
            density = kde(support)

        return density, support

    def _eval_bivariate(self, x1, x2, weights=None):
        """Fit and evaluate on bivariate data."""
        support = self.support
        if support is None:
            support = self.define_support(x1, x2, cache=False)

        kde = self._fit([x1, x2], weights)

        if self.cumulative:
            grid1, grid2 = support
            density = np.zeros((grid1.size, grid2.size))
            p0 = min(grid1), min(grid2)
            for i, xi in enumerate(grid1):
                for j, xj in enumerate(grid2):
                    density[i, j] = kde.integrate_box(p0, (xi, xj))

        else:
            xx1, xx2 = np.meshgrid(*support)
            density = kde([xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

        return density, support

    def __call__(self, x1, x2=None, weights=None):
        """Fit and evaluate on univariate or bivariate data."""
        if x2 is None:
            return self._eval_univariate(x1, weights)
        else:
            return self._eval_bivariate(x1, x2, weights)


def univariate_density(
        data_variable,
        estimate_kws,
):
    # Initialize the estimator object
    estimator = KDE(**estimate_kws)

    all_data = data_variable.dropna()

    # Extract the data points from this sub set and remove nulls
    sub_data = all_data.dropna()
    observations = sub_data[data_variable]

    observation_variance = observations.var()
    if math.isclose(observation_variance, 0) or np.isnan(observation_variance):
        msg = "Dataset has 0 variance; skipping density estimate."
        warnings.warn(msg, UserWarning)

    # Estimate the density of observations at this level
    density, support = estimator(observations, weights=None)

    return density, support


def bivariate_density(
        data: pd.DataFrame,
        estimate_kws: dict,
):
    """
    Estimate bivariate KDE
    :param data: DataFrame containing (x, y) data
    :param estimate_kws: KDE parameters
    :return:
    """

    estimator = KDE(**estimate_kws)

    all_data = data.dropna()

    # Extract the data points from this sub set and remove nulls
    sub_data = all_data.dropna()
    observations = sub_data[["x", "y"]]

    # Check that KDE will not error out
    variance = observations[["x", "y"]].var()
    if any(math.isclose(x, 0) for x in variance) or variance.isna().any():
        msg = "Dataset has 0 variance; skipping density estimate."
        warnings.warn(msg, UserWarning)

    # Estimate the density of observations at this level
    observations = observations["x"], observations["y"]
    density, support = estimator(*observations, weights=None)

    # Transform the support grid back to the original scale
    xx, yy = support

    support = xx, yy

    return density, support


def kde_params(
        x: np.array = None,
        y: np.array = None,
        bw: float = None,
        gridsize: int = 200,
        cut: float = 3, clip=None, cumulative: bool = False,
        bw_method: str = "scott", bw_adjust: int = 1
):
    """
    Obtain density and support (grid) of the bivariate KDE
    :param x:
    :param y:
    :param bw:
    :param gridsize:
    :param cut:
    :param clip:
    :param cumulative:
    :param bw_method:
    :param bw_adjust:
    :return:
    """

    data = {'x': x, 'y': y}
    frame = pd.DataFrame(data=data)

    # Handle deprecation of `bw`
    if bw is not None:
        bw_method = bw

    # Pack the kwargs for statistics.KDE
    estimate_kws = dict(
        bw_method=bw_method,
        bw_adjust=bw_adjust,
        gridsize=gridsize,
        cut=cut,
        clip=clip,
        cumulative=cumulative,
    )

    if y is None:
        density, support = univariate_density(
            data_variable=frame,
            estimate_kws=estimate_kws
        )

    else:
        density, support = bivariate_density(
            data=frame,
            estimate_kws=estimate_kws,
        )

    return density, support


def pixel_coordinate(line: list,
                     x_1d: np.array,
                     y_1d: np.array):
    """
    Gets the pixel coordinate of the value x or y, in order to get posterior conditional probability given a KDE.
    :param line: Coordinates of the line we'd like to sample along [(x1, y1), (x2, y2)]
    :param x_1d: List of x coordinates along the axis
    :param y_1d: List of y coordinates along the axis
    :return:
    """
    # https://stackoverflow.com/questions/18920614/plot-cross-section-through-heat-map
    # Convert the line to pixel/index coordinates
    x_world, y_world = np.array(list(zip(*line)))
    col = y_1d.shape * (x_world - min(x_1d)) / x_1d.ptp()
    row = x_1d.shape * (y_world - min(y_1d)) / y_1d.ptp()

    # Interpolate the line at "num" points...
    num = 200
    row, col = [np.linspace(item[0], item[1], num) for item in [row, col]]

    return row, col


def conditional_distribution(kde_array: np.array,
                             x_array: np.array,
                             y_array: np.array,
                             x: float = None,
                             y: float = None
                             ):
    """
    Compute the conditional posterior distribution p(x_array|y_array) given x or y.
    Provide only one observation ! Either x or y.
    Perform a cross-section in the KDE along the y axis.
    :param x: Observed data (horizontal axis)
    :param y: Observed data (vertical axis)
    :param kde_array: KDE of the prediction
    :param x_array: X grid (1D)
    :param y_array: Y grid (1D)
    :return:
    """

    # Coordinates of the line we'd like to sample along
    if x is not None:
        line = [(x, min(y_array)), (x, max(y_array))]
    elif y is not None:
        line = [(min(x_array), y), (max(x_array), y)]
    else:
        msg = "No observation point included."
        warnings.warn(msg, UserWarning)
        return 0

    # Convert line to row/column
    row, col = pixel_coordinate(line=line, x_1d=x_array, y_1d=y_array)

    # Extract the values along the line, using cubic interpolation
    zi = ndimage.map_coordinates(kde_array, np.vstack((row, col)))

    return zi


def normalize_distribution(post: np.array,
                           support: np.array):
    """
    When a cross-section is performed along a bivariate KDE, the integral might not = 1.
    This function normalizes such functions so that their integral = 1.
    :param post: Values of the KDE cross-section
    :param support: Corresponding support
    :return:
    """
    a = integrate.simps(y=np.abs(post), x=support)

    if np.abs(a - 1) > 1e-4:  # Rule of thumb
        post *= 1 / a

    return post


def posterior_conditional(x: np.array,
                          y: np.array,
                          x_obs: float = None,
                          y_obs: float = None):
    """
    Computes the posterior distribution p(y|x_obs) or p(x|y_obs) by doing a cross section of the KDE of (d, h).
    :param x: Predictor (x-axis)
    :param y: Target (y-axis)
    :param x_obs: Observation (predictor, x-axis)
    :param y_obs: Observation (target, y-axis)
    :return:
    """
    # Compute KDE
    dens, support = kde_params(x=x, y=y)
    # Grid parameters
    xg, yg = support

    if x_obs is not None:
        support = yg
        # Extract the density values along the line, using cubic interpolation
        post = conditional_distribution(x=x_obs,
                                        x_array=xg,
                                        y_array=yg,
                                        kde_array=dens)
    elif y_obs is not None:
        support = xg
        # Extract the density values along the line, using cubic interpolation
        post = conditional_distribution(y=y_obs,
                                        x_array=xg,
                                        y_array=yg,
                                        kde_array=dens)

    else:
        msg = "No observation point included."
        warnings.warn(msg, UserWarning)
        return 0

    post = normalize_distribution(post, support)

    return post, support


class PosteriorIO:
    """
    Heart of the framework.
    """
    def __init__(self, directory: str = None):
        self.posterior_mean = None
        self.posterior_covariance = None
        self.seed = None
        self.n_posts = None
        self.normalize_h = PowerTransformer(method='yeo-johnson', standardize=True)
        self.normalize_d = PowerTransformer(method='yeo-johnson', standardize=True)
        self.directory = directory

    def mvn_inference(self,
                      h_cca_training_gaussian,
                      d_cca_training,
                      d_pc_training,
                      d_rotations,
                      d_cca_prediction):
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
            shctg = np.shape(h_cca_training_gaussian)  # Shape = (n_components_CCA, n_training)
        if isinstance(d_cca_training, (list, tuple, np.ndarray)):
            sdct = np.shape(d_cca_training)  # Shape = (n_components_CCA, n_training)
        if isinstance(d_pc_training, (list, tuple, np.ndarray)):
            sdpt = np.shape(d_pc_training)  # Shape = (n_training, n_components_PCA)
        if isinstance(d_rotations, (list, tuple, np.ndarray)):
            sdr = np.shape(d_rotations)  # Shape = (n_components_PCA_d, n_components_CCA_h)
        if isinstance(d_cca_prediction, (list, tuple, np.ndarray)):
            sdcp = np.shape(d_cca_prediction)  # Shape = (n_components_CCA, 1)

        # Size of the set
        n_training = d_cca_training.shape[0]

        # Computation of the posterior mean in Canonical space
        h_mean = np.mean(h_cca_training_gaussian, axis=0)  # (n_comp_CCA, 1)
        h_mean = np.where(np.abs(h_mean) < 1e-12, 0, h_mean)  # Mean is 0, as expected.

        # Evaluate the covariance in h (in Canonical space)
        # Very close to the Identity matrix
        h_cov_operator = np.cov(h_cca_training_gaussian.T)  # (n_comp_CCA, n_comp_CCA)

        # Evaluate the covariance in d (here we assume no data error, so C is identity times a given factor)
        x_dim = np.size(d_pc_training, axis=1)  # Number of PCA components for the curves
        noise = .01
        d_cov_operator = np.eye(x_dim) * noise  # I matrix. (n_comp_PCA, n_comp_PCA)
        d_noise_covariance = d_rotations.T @ d_cov_operator @ d_rotations  # (n_comp_CCA, n_comp_CCA)

        # Linear modeling h to d (in canonical space) with least-square criterion.
        # Pay attention to the transpose operator.
        # Computes the vector g that approximately solves the equation h @ g = d.
        g = np.linalg.lstsq(h_cca_training_gaussian, d_cca_training, rcond=None)[0].T
        # Replace values below threshold by 0.
        g = np.where(np.abs(g) < 1e-12, 0, g)  # (n_comp_CCA, n_comp_CCA)

        # Modeling error due to deviations from theory
        d_ls_predicted = h_cca_training_gaussian @ g.T  # (n_components_CCA, n_training)
        d_modeling_mean_error = np.mean(d_cca_training - d_ls_predicted, axis=0)  # (n_comp_CCA, 1)
        d_modeling_error = \
            d_cca_training \
            - d_ls_predicted \
            - np.tile(d_modeling_mean_error, (n_training, 1))
        # (n_comp_CCA, n_training)

        # Information about the covariance of the posterior distribution in Canonical space.
        d_modeling_covariance = np.cov(d_modeling_error.T)  # (n_comp_CCA, n_comp_CCA)

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
        h_posterior_mean = h_posterior_covariance @ \
                           (d11 @ h_mean - d12 @ (d_cca_prediction[0] - d_modeling_mean_error - h_mean @ g.T))

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
        self.posterior_covariance = h_posterior_covariance  # (n_comp_CCA, n_comp_CCA)

    def back_transform(self,
                       h_posts_gaussian,
                       cca_obj,
                       pca_h,
                       n_posts: int,
                       add_comp: bool = False,
                       save_target_pc: bool = False):
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
        h_posts = self.normalize_h.inverse_transform(h_posts_gaussian)  # (n_components, n_samples)

        # Calculate the values of hf, i.e. reverse the canonical correlation, it always works if dimf > dimh
        # The value of h_pca_reverse are the score of PCA in the forecast space.
        # To reverse data in the original space, perform the matrix multiplication between the data in the CCA space
        # with the y_loadings matrix. Because CCA scales the input, we must multiply the output by the y_std dev
        # and add the y_mean.
        h_pca_reverse = np.matmul(h_posts, cca_obj.y_loadings_.T) * cca_obj.y_std_ + cca_obj.y_mean_

        # Whether to add or not the rest of PC components
        if add_comp:  # TODO: double check
            rnpc = np.array([pca_h.pc_random(n_posts) for _ in range(n_posts)])  # Get the extra components
            h_pca_reverse = np.array([np.concatenate((h_pca_reverse[i], rnpc[i])) for i in range(n_posts)])  # Insert it

        if save_target_pc:
            fname = jp(self.directory, 'target_pc.npy')
            np.save(fname, h_pca_reverse)

        # Generate forecast in the initial dimension and reshape.
        forecast_posterior = \
            pca_h.custom_inverse_transform(h_pca_reverse).reshape((n_posts,
                                                                   pca_h.training_shape[1],
                                                                   pca_h.training_shape[2]))

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
        h_posts_gaussian = np.random.multivariate_normal(mean=self.posterior_mean,
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
            d_pc_training, d_pc_prediction = pca_d.pca_refresh(pca_d.n_pc_cut)
            h_pc_training, _ = pca_h.pca_refresh(pca_h.n_pc_cut)

            d_pc_obs = d_pc_prediction[0]  # observation data for prediction sample

            # Transform to canonical space
            d_cca_training, h_cca_training = transform(d_pc_training, h_pc_training)
            # d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T

            # Ensure Gaussian distribution in d_cca_training
            d_cca_training = self.normalize_d.fit_transform(d_cca_training)

            # Ensure Gaussian distribution in h_cca_training
            # h_cca_training_gaussian = self.processing.gaussian_distribution(h_cca_training)
            h_cca_training = self.normalize_h.fit_transform(h_cca_training)

            # Get the rotation matrices
            d_rotations = cca_obj.x_rotations_

            # Project observed data into canonical space.
            d_cca_prediction = transform(d_pc_obs.reshape(1, -1))

            d_cca_prediction = self.normalize_d.transform(d_cca_prediction)

            # Estimate the posterior mean and covariance (Tarantola)

            self.mvn_inference(h_cca_training,
                               d_cca_training,
                               d_pc_training,
                               d_rotations,
                               d_cca_prediction)

            # Set the seed for later use
            if self.seed is None:
                self.seed = np.random.randint(2 ** 32 - 1, dtype='uint32')

            if n_posts is None:
                self.n_posts = setup.forecast.n_posts
            else:
                self.n_posts = n_posts

            # Saves this postio object to avoid saving large amounts of 'forecast_posterior'
            # This allows to reload this object later on and resample using the same seed.
            joblib.dump(self, jp(self.directory, 'post.pkl'))

        # Sample the inferred multivariate gaussian distribution
        h_posts_gaussian = self.random_sample(self.n_posts)

        # Back-transform h posterior to the physical space
        forecast_posterior = self.back_transform(h_posts_gaussian=h_posts_gaussian,
                                                 cca_obj=cca_obj,
                                                 pca_h=pca_h,
                                                 n_posts=self.n_posts,
                                                 add_comp=add_comp)

        return forecast_posterior


def transform(f,
              k_mean: float,
              k_std: float):
    """
    Transforms the values of the statistical_simulation simulations into meaningful data.
    :param: f: np.array: Simulation output = Hk field
    :param: k_mean: float: Mean of the Hk field
    :param: k_std: float: Standard deviation of the Hk field
    """

    ff = f * k_std + k_mean

    return 10 ** ff


def sgsim(model_ws: str,
          grid_dir: str,
          wells_hk: list = None):
    """
    Perform sequential gaussian simulation to generate K fields.
    :param model_ws: str: Working directory
    :param grid_dir: str: Grid directory
    :param wells_hk: List[float]: K values at wells
    :return:
    """
    # %% Initiate sgems pjt
    pjt = sg.Sgems(project_name='sgsim', project_wd=grid_dir, res_dir=model_ws)

    # %% Load hard data point set

    data_dir = grid_dir
    dataset = 'wells.eas'
    file_path = jp(data_dir, dataset)

    hd = PointSet(project=pjt, pointset_path=file_path)

    if wells_hk is None:
        hku = 2. + np.random.rand(len(hd.dataframe))  # Fix hard data values at wells location
    else:
        hku = wells_hk

    if not os.path.exists(jp(model_ws, setup.files.sgems_file)):
        hd.dataframe['hd'] = hku
        hd.export_01('hd')  # Exports modified dataset in binary

    # %% Generate grid. Grid dimensions can automatically be generated based on the data points
    # unless specified otherwise, but cell dimensions dx, dy, (dz) must be specified
    gd = setup.grid_dimensions()
    Discretize(project=pjt, dx=gd.dx, dy=gd.dy, xo=gd.xo, yo=gd.yo, x_lim=gd.x_lim, y_lim=gd.y_lim)

    # Get sgems grid centers coordinates:
    x = np.cumsum(pjt.dis.along_r) - pjt.dis.dx / 2
    y = np.cumsum(pjt.dis.along_c) - pjt.dis.dy / 2
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
    centers = np.stack((xv, yv), axis=2).reshape((-1, 2))

    if os.path.exists(jp(model_ws, 'hk0.npy')):
        hk0 = np.load(jp(model_ws, 'hk0.npy'))
        return hk0, centers

    # %% Display point coordinates and grid
    # pl = Plots(project=pjt)
    # pl.plot_coordinates()

    # %% Load your algorithm xml file in the 'algorithms' folder.
    dir_path = os.path.abspath(__file__ + "/..")
    algo_dir = jp(dir_path, 'algorithms')
    al = XML(project=pjt, algo_dir=algo_dir)
    al.xml_reader('bel_sgsim')

    # %% Modify xml below:
    al.xml_update('Seed', 'value', str(np.random.randint(1e9)), show=0)

    # %% Write python script
    pjt.write_command()

    # %% Run sgems
    pjt.run()
    # Plot 2D results
    # pl.plot_2d(save=True)

    opl = jp(model_ws, 'results.grid')  # Output file location.

    matrix = data_read(opl, start=3)  # Grid information directly derived from the output file.
    matrix = np.where(matrix == -9966699, np.nan, matrix)

    k_mean = np.random.uniform(1.4, 2)  # Hydraulic conductivity exponent mean between x and y.
    print(f'hk mean={10 ** k_mean} m/d')
    k_std = 0.4  # Log value of the standard deviation

    tf = np.vectorize(transform)  # Transform values from log10
    matrix = tf(matrix, k_mean, k_std)  # Apply function to results

    matrix = matrix.reshape((pjt.dis.nrow, pjt.dis.ncol))  # reshape - assumes 2D !
    matrix = np.flipud(matrix)  # Flip to correspond to sgems

    # import matplotlib.pyplot as plt
    # extent = (pjt.dis.xo, pjt.dis.x_lim, pjt.dis.yo, pjt.dis.y_lim)
    # plt.imshow(np.log10(matrix), cmap='coolwarm', extent=extent)
    # plt.plot(pjt.point_set.raw_data[:, 0], pjt.point_set.raw_data[:, 1], 'k+', markersize=1, alpha=.7)
    # plt.colorbar()
    # plt.show()

    np.save(jp(model_ws, 'hk0'), matrix)  # Save the un-discretized hk grid

    return matrix, centers