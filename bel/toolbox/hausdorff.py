import numpy as np
from numpy.core.umath_tests import inner1d


# a = np.array([[1,2],[3,4],[5,6],[7,8]])
# b = np.array([[2,3],[4,5],[6,7],[8,9],[10,11]])
# https://github.com/sapphire008/Python/blob/master/generic/HausdorffDistance.py

def distance(a, b):
    # Hausdorff Distance: Compute the Hausdorff distance between two point
    # clouds.
    # Let a and b be subsets of metric space (Z,dZ),
    # The Hausdorff distance between a and b, denoted by dH(a,b),
    # is defined by:
    # dH(a,b) = max(h(a,b),h(b,a)),
    # where h(a,b) = max(min(d(a,b))
    # and d(a,b) is a L2 norm
    # dist_H = hausdorff(a,b)
    # a: First point sets (MxN, with M observations in N dimension)
    # b: Second point sets (MxN, with M observations in N dimension)
    # ** a and b may have different number of rows, but must have the same
    # number of columns.
    #
    # Edward DongBo Cui; Stanford University; 06/17/2014

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(a, a)[np.newaxis].T + inner1d(b, b) - 2 * (np.dot(a, b.T)))
    # Find DH
    dH = np.max(np.array([np.max(np.min(D_mat, axis=0)), np.max(np.min(D_mat, axis=1))]))
    return dH


def modified_distance(a, b):
    # This function computes the Modified Hausdorff Distance (MHD) which is
    # proven to function better than the directed HD as per Dubuisson et al.
    # in the following work:
    #
    # M. P. Dubuisson and a. K. Jain. a Modified Hausdorff distance for object
    # matching. In ICPR94, pages a:566-568, Jerusalem, Israel, 1994.
    # http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361
    #
    # The function computes the forward and reverse distances and outputs the
    # maximum/minimum of both.
    # Optionally, the function can return forward and reverse distance.
    #
    # Format for calling function:
    #
    # [MHD,FHD,RHD] = ModHausdorffDist(a,b);
    #
    # where
    # MHD = Modified Hausdorff Distance.
    # FHD = Forward Hausdorff Distance: minimum distance from all points of b
    #      to a point in a, averaged for all a
    # RHD = Reverse Hausdorff Distance: minimum distance from all points of a
    #      to a point in b, averaged for all b
    # a -> Point set 1, [row as observations, and col as dimensions]
    # b -> Point set 2, [row as observations, and col as dimensions]
    #
    # No. of samples of each point set may be different but the dimension of
    # the points must be the same.
    #
    # Edward DongBo Cui Stanford University; 06/17/2014

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(a, a)[np.newaxis].T + inner1d(b, b) - 2 * (np.dot(a, b.T)))
    # Calculating the forward HD: mean(min(each col))
    FHD = np.mean(np.min(D_mat, axis=1))
    # Calculating the reverse HD: mean(min(each row))
    RHD = np.mean(np.min(D_mat, axis=0))
    # Calculating mhd
    MHD = np.max(np.array([FHD, RHD]))

    return MHD, FHD, RHD
