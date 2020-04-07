from os.path import join as jp

import numpy as np
import skfmm  # Library to compute the signed distance
from matplotlib.patches import Polygon


def get_centroids(array, grf):
    xys = np.dstack((np.flip((np.indices(array.shape) + 1), 0) * grf - grf / 2))  # Getting centroids
    return xys.reshape((array.shape[0] * array.shape[1], 2))


class SignedDistance:

    def __init__(self, x_lim=1500, y_lim=1000, grf=1):
        self.grf = grf
        self.nrow = int(y_lim / grf)
        self.ncol = int(x_lim / grf)
        array = np.ones((self.nrow, self.ncol))
        self.xys = get_centroids(array, grf)

    def matrix_poly_bin(self, pzs, outside=-1, inside=1):
        poly = Polygon(pzs, True)  # Creates a Polygon abject out of the polygon vertices in pzs
        ind = np.nonzero(poly.contains_points(self.xys))[0]  # Checks which points are enclosed by polygon.
        phi = np.ones((self.nrow, self.ncol))*outside  # SD - create matrix of -1
        phi[ind] = inside  # Points inside the WHPA are assigned a value of 1, and 0 for those outside
        return phi

    def function(self, pzs):

        phi = self.matrix_poly_bin(pzs)

        sd = skfmm.distance(phi, dx=self.grf)  # Signed distance computation

        return sd
