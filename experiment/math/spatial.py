#  Copyright (c) 2021. Robin Thibaut, Ghent University

import numpy as np
import skfmm  # Library to compute the signed distance
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial.distance import cdist


def get_centroids(array,
                  grf: float):
    """
    Given a (m, n) matrix of cells dimensions in the x-y axes, returns the (m, n, 2) matrix of the coordinates of
    centroids.
    :param array: (m, n) array
    :param grf: float: Cell dimension
    """
    xys = np.dstack((np.flip((np.indices(array.shape) + 1), 0) * grf - grf / 2))  # Getting centroids
    return xys.reshape((array.shape[0] * array.shape[1], 2))


class Spatial:

    def __init__(self,
                 x_lim: float = None,
                 y_lim: float = None,
                 grf: float = 1):
        if y_lim is None:
            y_lim = [0, 1000]
        if x_lim is None:
            x_lim = [0, 1500]
        self.grf = grf  # Cell dimension
        self.nrow = int(np.diff(y_lim) / grf)  # Number of rows
        self.ncol = int(np.diff(x_lim) / grf)  # Number of columns
        array = np.ones((self.nrow, self.ncol))  # Dummy array
        self.xys = get_centroids(array, grf) + np.min([x_lim, y_lim], axis=1)  # Centroids of dummy array

    def matrix_poly_bin(self,
                        pzs,
                        outside: int = -1,
                        inside: int = 1):
        """
        Given a polygon whose vertices are given by the array pzs, and a matrix of
        centroids coordinates of the surface discretization, assigns to the matrix a certain value
        whether the cell is inside or outside said polygon.

        To compute the signed distance function, we need a negative/positive value.

        :param pzs: Polygon vertices (v, 2)
        :param outside: Value to assign to the matrix outside of the polygon
        :param inside: Value to assign to the matrix inside of the polygon
        :return: phi = the binary matrix
        """

        poly = Polygon(pzs, True)  # Creates a Polygon abject out of the polygon vertices in pzs
        ind = np.nonzero(poly.contains_points(self.xys))[0]  # Checks which points are enclosed by polygon.
        phi = np.ones((self.nrow, self.ncol))*outside  # SD - create matrix of 'outside'
        phi = phi.reshape((self.nrow * self.ncol))  # Flatten to have same dimension as 'ind'
        phi[ind] = inside  # Points inside the WHPA are assigned a value of 'inside'
        phi = phi.reshape((self.nrow, self.ncol))  # Reshape

        return phi

    def signed_distance(self, pzs):
        """
        Given an array of coordinates of polygon vertices, computes its signed distance field.
        :param pzs: Array of ordered vertices coordinates of a polygon.
        :return: Signed distance matrix
        """

        phi = self.matrix_poly_bin(pzs)

        sd = skfmm.distance(phi, dx=self.grf)  # Signed distance computation

        return sd


def contours_vertices(x, y,
                      arrays,
                      c=0):
    """
    Extracts contour vertices from a list of matrices
    :param arrays: list of matrices
    :param c: Contour value
    :return: vertices array
    """
    if len(arrays.shape) < 3:
        arrays = [arrays]
    # First create figures for each forecast.
    figs = [plt.figure() for _ in range(len(arrays))]
    c0s = [plt.contour(x, y, f, [c]) for f in arrays]
    [plt.close(f) for f in figs]  # Close plots
    # .allseg[0][0] extracts the vertices of each O contour = WHPA's vertices
    v = np.array([c0.allsegs[0][0] for c0 in c0s], dtype=object)
    return v


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