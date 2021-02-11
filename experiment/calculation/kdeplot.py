#  Copyright (c) 2021. Robin Thibaut, Ghent University

import math
import warnings
import pandas as pd
import numpy as np
from scipy import stats, ndimage

import matplotlib.pyplot as plt


class KDE:
    """Bivariate kernel density estimator."""

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
        """Fit the scipy kde while adding bw_adjust logic and version check."""
        fit_kws = {"bw_method": self.bw_method}
        if weights is not None:
            fit_kws["weights"] = weights

        kde = stats.gaussian_kde(fit_data, **fit_kws)
        kde.set_bandwidth(kde.factor * self.bw_adjust)

        return kde

    def _eval_univariate(self, x, weights=None):
        """Fit and evaluate a univariate on univariate data."""
        support = self.support
        if support is None:
            support = self.define_support(x, cache=False)

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
        """Fit and evaluate a univariate on bivariate data."""
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
        data,
        estimate_kws,
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
        x=None,
        y=None,
        bw=None,
        gridsize=200,
        cut=3, clip=None, cumulative=False,
        bw_method="scott", bw_adjust=1
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
    :param multiple:
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


def pixel_coordinate(x: float,
                     x_1d: np.array,
                     y_1d: np.array):
    """
    Gets the pixel coordinate of the value x, in order to get posterior conditional probability given a KDE.
    :param x: Value along the horizontal axis
    :param x_1d: List of x coordinates along the axis
    :param y_1d: List of y coordinates along the axis
    :return:
    """

    n_pix_y = len(y_1d)  # pixels in y-axis

    x_index = np.argmin(np.abs(x_1d - x))  # pixel index of x value

    # Extract the line (pixel coordinates)
    # Limits:
    xo, yo = x_index, 0
    xf, yf = x_index, n_pix_y
    #
    num = len(y_1d)
    x_, y_ = np.ones(num)*x_index, np.linspace(yo, yf, num)

    return x_, y_


def conditional_distribution(x: float,
                             y_kde: np.array,
                             x_array: np.array,
                             y_array: np.array,
                             ):
    """
    Compute the conditional posterior distribution p(x|y) given d.
    :param x: Observed data
    :param y_kde: KDE of the prediction
    :param x_array: X grid (1D)
    :param y_array: Y grid (1D)
    :return:
    """
    x_i, y_i = pixel_coordinate(x, x_array, y_array)
    # Extract the density values along the line, using cubic interpolation
    zi = ndimage.map_coordinates(y_kde, np.vstack((x_i, y_i)))
    return zi


def posterior_conditional(d, h, d_c):
    # Compute KDE
    dens, sup = kde_params(x=d, y=h)
    # Grid parameters
    xg, yg = sup
    sup = yg
    # Extract the density values along the line, using cubic interpolation
    post = conditional_distribution(x=d_c,
                                    x_array=xg,
                                    y_array=yg,
                                    y_kde=dens)
    return post, sup


if __name__ == '__main__':
    # Generate some 2D data
    rs = np.random.RandomState(5)
    mean = [0, 0]
    cov = [(1, .98), (.98, 1)]
    predictor, target = rs.multivariate_normal(mean, cov, 200).T

    conditional_value = 0
    h_post, sup = posterior_conditional(predictor, target, conditional_value)

    plt.plot(h_post)
    plt.show()
