#  Copyright (c) 2021. Robin Thibaut, Ghent University

import numpy as np
import skfmm  # Library to compute the signed distance
from matplotlib.patches import Polygon


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


class SignedDistance:

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

    def compute(self, pzs):
        """
        Given an array of coordinates of polygon vertices, computes its signed distance field.
        :param pzs: Array of ordered vertices coordinates of a polygon.
        :return: Signed distance matrix
        """

        phi = self.matrix_poly_bin(pzs)

        sd = skfmm.distance(phi, dx=self.grf)  # Signed distance computation

        return sd
