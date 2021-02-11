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

    def _eval_bivariate(self, x1, x2, weights=None):
        """Fit and evaluate a univariate on bivariate data."""
        support = self.support
        if support is None:
            support = self.define_support(x1, x2, cache=False)

        kde = self._fit([x1, x2], weights)

        if self.cumulative:

            grid1, grid2 = support
            density = np.zeros((grid1.size, grid2.size))
            p0 = grid1.min(), grid2.min()
            for i, xi in enumerate(grid1):
                for j, xj in enumerate(grid2):
                    density[i, j] = kde.integrate_box(p0, (xi, xj))

        else:

            xx1, xx2 = np.meshgrid(*support)
            density = kde([xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

        return density, support

    def __call__(self, x1, x2, weights=None):
        """Fit and evaluate on univariate or bivariate data."""
        return self._eval_bivariate(x1, x2, weights)


def bivariate_density(
        data,
        estimate_kws,
):
    """
    
    :param data: 
    :param estimate_kws: 
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


def kdeplot(
        x=None,
        y=None,
        bw=None,
        gridsize=200,
        cut=3, clip=None, cumulative=False,
        bw_method="scott", bw_adjust=1, log_scale=None,
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
    :param log_scale: 
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

    density, support = bivariate_density(
        data=frame,
        estimate_kws=estimate_kws,
    )

    return density, support


if __name__ == '__main__':
    rs = np.random.RandomState(5)
    mean = [0, 0]
    cov = [(1, .98), (.98, 1)]
    d, h = rs.multivariate_normal(mean, cov, 200).T

    dens, sup = kdeplot(x=d, y=h)

    xg, yg = sup

    plt.imshow(np.flipud(dens))
    plt.show()

    # -- Extract the line...
    # Make a line with "num" points...
    x0, y0 = 100, 0  # These are in _pixel_ coordinates!!
    x1, y1 = 100, 200
    num = 200
    x_, y_ = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

    # Extract the values along the line, using cubic interpolation
    zi = ndimage.map_coordinates(dens, np.vstack((x_, y_)))

    plt.plot(zi)
    plt.show()
