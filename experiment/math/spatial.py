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
            self.y_lim = [0, 1000]
        else:
            self.y_lim = y_lim
        if x_lim is None:
            self.x_lim = [0, 1500]
        else:
            self.x_lim = x_lim

        self.grf = grf  # Cell dimension
        self.nrow = int(np.diff(self.y_lim) / grf)  # Number of rows
        self.ncol = int(np.diff(self.x_lim) / grf)  # Number of columns
        array = np.ones((self.nrow, self.ncol))  # Dummy array
        self.xys = get_centroids(array, grf) + np.min([self.x_lim, self.y_lim], axis=1)  # Centroids of dummy array

    def binary_stack(self, vertices):
        """
        Takes WHPA vertices and binarizes the image (e.g. 1 inside, 0 outside WHPA).
        """
        # For this approach we use our SignedDistance module
        # Create binary images of WHPA stored in bin_whpa
        bin_whpa = [self.matrix_poly_bin(pzs=p, inside=1 / (len(vertices)-1), outside=0) for p in vertices]
        big_sum = np.sum(bin_whpa, axis=0)  # Stack them
        b_low = np.where(big_sum >= 1, 0, big_sum)  # Replace 1 values by 0
        # b_low = np.where(big_sum == 0, 1, big_sum)  # Replace 0 values by 1
        return b_low

    def matrix_poly_bin(self,
                        pzs,
                        outside: float = -1,
                        inside: float = 1):
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


def contours_vertices(x: list, y: list,
                      arrays: np.array,
                      c: float = 0,
                      ignore_: bool = True):
    """
    Extracts contour vertices from a list of matrices.
    :param x:
    :param y:
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
    if ignore_:
        v = np.array([c0.allsegs[0][0] for c0 in c0s], dtype=object)
    else:
        v = np.array([c0.allsegs[0][i] for c0 in c0s for i in range(len(c0.allsegs[0]))], dtype=object)
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
