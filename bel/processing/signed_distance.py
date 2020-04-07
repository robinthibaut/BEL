import numpy as np
import skfmm  # Library to compute the signed distance
from matplotlib.patches import Polygon


def get_centroids(array, grf):
    # Given a m x n matrix, returns the m x n x 2 matrix of the coordinates of centroids.
    xys = np.dstack((np.flip((np.indices(array.shape) + 1), 0) * grf - grf / 2))  # Getting centroids
    return xys.reshape((array.shape[0] * array.shape[1], 2))


class SignedDistance:

    def __init__(self, x_lim=1500, y_lim=1000, grf=1):
        self.grf = grf  # Cell dimension
        self.nrow = int(y_lim / grf)  # Number of rows
        self.ncol = int(x_lim / grf)  # Number of columns
        array = np.ones((self.nrow, self.ncol))  # Dummy array
        self.xys = get_centroids(array, grf)  # Centroids of dummy array

    def matrix_poly_bin(self, pzs, outside=-1, inside=1):
        """
        Given a polygon whose vertices are given by the array pzs, and a matrix of
        centroids coordinates of the surface discretization, assign to the matrix a certain value
        whether the cell is inside or outside said polygon.
        To compute the signed distance function, we need a negative/positive value.

        @param pzs: Polygon vertices
        @param outside: Value to assign to the matrix outside of the polygon
        @param inside: Value to assign to the matrix inside of the polygon
        @return: phi, the binary matrix
        """
        poly = Polygon(pzs, True)  # Creates a Polygon abject out of the polygon vertices in pzs
        ind = np.nonzero(poly.contains_points(self.xys))[0]  # Checks which points are enclosed by polygon.
        phi = np.ones((self.nrow, self.ncol))*outside  # SD - create matrix of -1
        phi = phi.reshape((self.nrow * self.ncol))
        phi[ind] = inside  # Points inside the WHPA are assigned a value of 1, and 0 for those outside
        phi = phi.reshape((self.nrow, self.ncol))
        return phi

    def function(self, pzs):

        phi = self.matrix_poly_bin(pzs)

        sd = skfmm.distance(phi, dx=self.grf)  # Signed distance computation

        return sd
