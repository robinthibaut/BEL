#  Copyright (c) 2020. Robin Thibaut, Ghent University

import numpy as np
from scipy.spatial.distance import cdist


def modified_distance(a, b):
    """
    Compute the modified Hausdorff distance between two N-D arrays.
    Distances between pairs are calculated using a Euclidean metric.

    Parameters
    ----------
    a : (M,N) ndarray
        Input array.
    b : (O,N) ndarray
        Input array.

    Returns
    -------
    d : double
        The modified Hausdorff distance between arrays `a` and `b`,

    Raises
    ------
    ValueError
        An exception is thrown if `u` and `v` do not have
        the same number of columns.

    References
    ----------
    .. [1] M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
           matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
           http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361

    See Also
    --------
    https://github.com/sapphire008/Python/blob/master/generic/HausdorffDistance.py

    """

    a = np.asarray(a, dtype=np.float64, order='c')
    b = np.asarray(b, dtype=np.float64, order='c')

    if a.shape[1] != b.shape[1]:
        raise ValueError('a and b need to have the same '
                         'number of columns')

    d = cdist(a, b)
    fhd = np.mean(np.min(d, axis=0))
    rhd = np.mean(np.min(d, axis=1))

    return max(fhd, rhd)
