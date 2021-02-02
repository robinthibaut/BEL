#  Copyright (c) 2021. Robin Thibaut, Ghent University

import numpy as np
import skfmm  # Library to compute the signed distance
from scipy.spatial.distance import cdist

from experiment.spatial.grid import get_centroids, binary_polygon


def grid_parameters(x_lim: list = None,
                    y_lim: list = None,
                    grf: float = 1):
    if y_lim is None:
        y_lim = [0, 1000]
    else:
        y_lim = y_lim
    if x_lim is None:
        x_lim = [0, 1500]
    else:
        x_lim = x_lim

    grf = grf  # Cell dimension
    nrow = int(np.diff(y_lim) / grf)  # Number of rows
    ncol = int(np.diff(x_lim) / grf)  # Number of columns
    array = np.ones((nrow, ncol))  # Dummy array
    xys = get_centroids(array, grf) + np.min([x_lim, y_lim], axis=1)  # Centroids of dummy array

    return xys, nrow, ncol


def signed_distance(xys: np.array,
                    nrow: int,
                    ncol: int,
                    grf: float,
                    pzs: np.array):
    """
    Given an array of coordinates of polygon vertices, computes its signed distance field.
    :param xys:
    :param nrow:
    :param ncol:
    :param grf:
    :param pzs: Array of ordered vertices coordinates of a polygon.
    :return: Signed distance matrix
    """

    phi = binary_polygon(xys, nrow, ncol, pzs)

    sd = skfmm.distance(phi, dx=grf)  # Signed distance computation

    return sd


def modified_hausdorff(a, b):
    """
    Compute the modified Hausdorff distance between two N-D arrays.
    Distances between pairs are calculated using an Euclidean metric.

    .. [1] M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
           matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
           http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361

    :param a : (M,N) ndarray. Input array.
    :param b : (O,N) ndarray. Input array.

    :return: d : double. The modified Hausdorff distance between arrays `a` and `b`.

    :raise ValueError: An exception is thrown if `a` and `b` do not have the same number of columns.

    Another Python implementation:
    https://github.com/sapphire008/Python/blob/master/generic/HausdorffDistance.py

    """

    a = np.asarray(a, dtype=np.float64, order='c')
    b = np.asarray(b, dtype=np.float64, order='c')

    if a.shape[1] != b.shape[1]:
        raise ValueError('a and b must have the same number of columns')

    d = cdist(a, b)  # Compute distance between each pair of the two collections of inputs.
    # dim(d) = (M, O)
    fhd = np.mean(np.min(d, axis=0))  # Mean of minimum values along rows
    rhd = np.mean(np.min(d, axis=1))  # Mean of minimum values along columns

    return max(fhd, rhd)


if __name__ == '__main__':
    lol = grid_parameters(x_lim=[0, 1500], y_lim=[0, 1000], grf=4)
