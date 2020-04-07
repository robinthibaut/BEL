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

    def function(self, pzs, results_dir=None):

        poly = Polygon(pzs, True)

        ind = np.nonzero(poly.contains_points(self.xys))[0]  # Checks which points are enclosed by polygon.

        phi = np.ones((self.nrow, self.ncol))*-1
        phi[ind] = 1  # Points inside the WHPA are assigned a value of 1, and 0 for those outside

        sd = skfmm.distance(phi, dx=self.grf)  # Signed distance computation

        if results_dir:
            np.save(jp(results_dir, 'sd'), sd)  # Save the array

        return sd

# def function(pzs,
#        grf=1,
#        x_lim=1500,
#        y_lim=1000,
#        results_dir=None):
#     # pz =  x-y coordinates endpoints particles
#     # delineation = tsp(pz)  # indices of the vertices of the final protection zone using TSP algorithm
#     # pzs = pz[delineation]  # x-y coordinates protection zone
#     # np.save(jp(results_dir, 'pz'), pzs)
#
#     # Points locations density
#     # from scipy.stats import gaussian_kde
#     # x = pz[:, 0]
#     # y = pz[:, 1]
#     # Calculate the point density
#     # xy = np.vstack([x, y])
#     # z = gaussian_kde(xy)(xy)
#     # fig, ax = plt.subplots()
#     # ax.scatter(x, y, c=z, s=100, edgecolor='')
#     # plt.show()
#
#     # Polygon approach - best
#
#     poly = Polygon(pzs, True)
#
#     nrow = int(y_lim / grf)
#     ncol = int(x_lim / grf)
#     phi = np.ones((nrow, ncol)) * -1  # Assign -1 to the matrix
#     xys = np.dstack((np.flip((np.indices(phi.shape) + 1), 0) * grf - grf / 2))  # Getting centroids
#
#     xys = xys.reshape((nrow * ncol, 2))
#     ind = np.nonzero(poly.contains_points(xys))[0]  # Checks which points are enclosed by polygon.
#     phi = phi.reshape((nrow * ncol))
#     phi[ind] = 1  # Points inside the WHPA are assigned a value of 1, and 0 for those outside
#     phi = phi.reshape((nrow, ncol))
#
#     sd = skfmm.distance(phi, dx=grf)  # Signed distance computation
#     if results_dir:
#         np.save(jp(results_dir, 'sd'), sd)  # Save the array
#
#     return sd
#
