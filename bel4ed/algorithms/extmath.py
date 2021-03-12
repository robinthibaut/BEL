import numpy as np

__all__ = ["svd_flip", "safe_accumulator_op", "signed_distance", "get_block", "matrix_paste", "h_sub"]

import skfmm

from scipy.spatial import distance_matrix

from bel4ed.spatial import block_shaped, binary_polygon


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u : ndarray
        u and v are the output of `linalg.svd`
        dimensions so one can compute `np.dot(u * s, v)`.

    v : ndarray
        u and v are the output of `linalg.svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.

    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.

    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


# Use at least float64 for the accumulating functions to avoid precision issue
# see https://github.com/numpy/numpy/issues/9393. The float64 is also retained
# as it is in case the float overflows
def safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.

    Parameters
    ----------
    op : function
        A numpy accumulator function such as np.mean or np.sum
    x : numpy array
        A numpy array to apply the accumulator function
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function

    Returns
    -------
    result : The output of the accumulator function passed to this function
    """
    if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=np.float64)
    else:
        result = op(x, *args, **kwargs)
    return result


def get_block(pm: np.array, i: int) -> np.array:
    """
    Extracts block from a 2x2 partitioned matrix.
    :param pm: Partitioned matrix
    :param i: Block index
    1 2
    3 4
    :return: Bock #b
    """

    b = pm.shape[0] // 2

    if i == 1:
        return pm[:b, :b]
    if i == 2:
        return pm[:b, b:]
    if i == 3:
        return pm[b:, :b]
    if i == 4:
        return pm[b:, b:]
    else:
        return 0


def matrix_paste(c_big: np.array, c_small: np.array) -> list:
    # Compute distance matrix between refined and dummy grid.
    dm = distance_matrix(c_big, c_small)
    inds = [
        np.unravel_index(np.argmin(dm[i], axis=None), dm[i].shape)[0]
        for i in range(dm.shape[0])
    ]
    return inds


def h_sub(h: np.array, un: int, uc: int, sc: int) -> np.array:
    """
    Process signed distance array.
    :param h: Signed distance array
    :param un: # rows
    :param uc: # columns
    :param sc: New cell dimension in x and y direction (original is 1)

    """
    h_u = np.zeros((h.shape[0], un, uc))
    for i in range(h.shape[0]):
        sim = h[i]
        sub = block_shaped(arr=sim, nrows=sc, ncols=sc)
        h_u[i] = np.array([s.mean() for s in sub]).reshape(un, uc)

    return h_u


def signed_distance(xys: np.array, nrow: int, ncol: int, grf: float, pzs: np.array):
    """
    Given an array of coordinates of polygon vertices, computes its signed distance field.
    :param xys: Centroids of a grid' cells
    :param nrow: Number of rows
    :param ncol: Number of columns
    :param grf: Grid dimension (uniform grid)
    :param pzs: Array of ordered vertices coordinates of a polygon.
    :return: Signed distance matrix
    """

    phi = binary_polygon(xys, nrow, ncol, pzs)

    sd = skfmm.distance(phi, dx=grf)  # Signed distance computation

    return sd