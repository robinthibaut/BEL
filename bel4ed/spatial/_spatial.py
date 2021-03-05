#  Copyright (c) 2021. Robin Thibaut, Ghent University
from typing import List

import numpy as np
import skfmm  # Library to compute the signed distance
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

__all__ = ['grid_parameters', 'signed_distance', 'modified_hausdorff', 'block_shaped', 'refine_axis', 'rc_from_blocks',
           'blocks_from_rc', 'blocks_from_rc_3d', 'matrix_paste', 'h_sub', 'get_centroids', 'contours_vertices',
           'binary_polygon', 'binary_stack', 'get_block', 'refine_machine']


def grid_parameters(x_lim: list = None, y_lim: list = None, grf: float = 1):
    """
    Generates grid parameters necessary prior to SD calculation.
    :param x_lim:
    :param y_lim:
    :param grf:
    :return:
    """
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
    # Centroids of dummy array
    xys = get_centroids(array, grf) + np.min([x_lim, y_lim], axis=1)

    return xys, nrow, ncol


def signed_distance(xys: np.array, nrow: int, ncol: int, grf: float,
                    pzs: np.array):
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

    a = np.asarray(a, dtype=np.float64, order="c")
    b = np.asarray(b, dtype=np.float64, order="c")

    if a.shape[1] != b.shape[1]:
        raise ValueError("a and b must have the same number of columns")

    # Compute distance between each pair of the two collections of inputs.
    d = cdist(a, b)
    # dim(d) = (M, O)
    fhd = np.mean(np.min(d, axis=0))  # Mean of minimum values along rows
    rhd = np.mean(np.min(d, axis=1))  # Mean of minimum values along columns

    return max(fhd, rhd)


def block_shaped(arr: np.array, nrows: int, ncols: int):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n sub-blocks with
    each sub-block preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisible by {}".format(
        h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisible by {}".format(
        w, ncols)

    return (arr.reshape(h // nrows, nrows, -1,
                        ncols).swapaxes(1, 2).reshape(-1, nrows, ncols))


def refine_axis(widths: List[float], r_pt: float, ext: float, cnd: float,
                d_dim: float, a_lim: float):
    """
    Refines one 1D axis around a point belonging to it.

    Example:
    along_c = refine_axis([10m, 10m... 10m], 500m, 70m, 2m, 10m, 1500m)

    :param widths: Array of cell widths along axis.
    :param r_pt: 1D point on the axis around which refining will occur.
    :param ext: Extent (distance) of the refinement around the point.
    :param cnd: New cell size after refinement.
    :param d_dim: Base cell dimension.
    :param a_lim: Limit of the axis.
    :return: Refined axis (widths)
    """

    x0 = widths
    x0s = np.cumsum(x0)  # Cumulative sum of the width of the cells
    pt = r_pt  # Point around which refining
    extx = ext  # Extent around the point
    cdrx = cnd
    dx = d_dim
    xlim = a_lim

    # X range of the polygon
    xrp = [pt - extx, pt + extx]

    # Where to refine
    wherex = np.where((xrp[0] < x0s) & (x0s <= xrp[1]))[0]

    # The algorithm must choose a 'flexible parameter', either the cell grid size, the dimensions of the grid or the
    # refined cells themselves... We choose to adapt the dimensions of the grid.
    exn = np.sum(x0[wherex])  # x-extent of the refinement zone
    fx = exn / cdrx  # divides the extent by the new cell spacing
    rx = exn % cdrx  # remainder
    if rx == 0:
        nwxs = np.ones(int(fx)) * cdrx
        x0 = np.delete(x0, wherex)
        x0 = np.insert(x0, wherex[0], nwxs)
    else:  # If the cells can not be exactly subdivided into the new cell dimension
        nwxs = np.ones(int(round(fx))) * cdrx  # Produce a new width vector
        x0 = np.delete(x0, wherex)  # Delete old cells
        x0 = np.insert(x0, wherex[0], nwxs)  # insert new

        # Cumulative width should equal x_lim, but it will not be the case, we have to adapt widths.
        cs = np.cumsum(x0)
        difx = xlim - cs[-1]
        # Location of cells whose widths will be adapted
        where_default = np.where(abs(x0 - dx) <= 5)[0]
        # Where do we have the default cell size on
        where_left = where_default[np.where(where_default < wherex[0])]
        # the left
        where_right = where_default[np.where(
            (where_default >= wherex[0] + len(nwxs)))]  # And on the right
        lwl = len(where_left)
        lwr = len(where_right)

        if lwl > lwr:
            rl = (
                    lwl / lwr
            )  # Weights how many cells are on either sides of the refinement zone
            # Splitting the extra widths on the left and right of the cells
            dal = difx / ((lwl + lwr) / lwl)
            dal = dal + (difx - dal) / rl
            dar = difx - dal
        elif lwr > lwl:
            rl = (
                    lwr / lwl
            )  # Weights how many cells are on either sides of the refinement zone
            # Splitting the extra widths on the left and right of the cells
            dar = difx / ((lwl + lwr) / lwr)
            dar = dar + (difx - dar) / rl
            dal = difx - dar
        else:
            # Splitting the extra widths on the left and right of the cells
            dal = difx / ((lwl + lwr) / lwl)
            dar = difx - dal

        x0[where_left] = x0[where_left] + dal / lwl
        x0[where_right] = x0[where_right] + dar / lwr

    return x0


def rc_from_blocks(blocks: np.array):
    """
    Computes the x and y dimensions of each block
    :param blocks:
    :return:
    """
    dc = np.array([np.diff(b[:, 0]).max for b in blocks])
    dr = np.array([np.diff(b[:, 1]).max for b in blocks])

    return dc, dr


def blocks_from_rc_3d(rows: np.array, columns: np.array):
    """
    Returns the blocks forming a 2D grid whose rows and columns widths are defined by the two arrays rows, columns
    """

    nrow = len(rows)
    ncol = len(columns)
    delr = rows
    delc = columns
    r_sum = np.cumsum(delr)
    c_sum = np.cumsum(delc)

    blocks = []
    for c in range(nrow):
        for n in range(ncol):
            b = [
                [c_sum[n] - delc[n], r_sum[c] - delr[c], 0.0],
                [c_sum[n] - delc[n], r_sum[c], 0.0],
                [c_sum[n], r_sum[c], 0.0],
                [c_sum[n], r_sum[c] - delr[c], 0.0],
            ]
            blocks.append(b)
    blocks = np.array(blocks)

    return blocks


def blocks_from_rc(rows: np.array, columns: np.array):
    """
    Returns the blocks forming a 2D grid whose rows and columns widths are defined by the two arrays rows, columns
    """

    nrow = len(rows)
    ncol = len(columns)
    delr = rows
    delc = columns
    r_sum = np.cumsum(delr)
    c_sum = np.cumsum(delc)

    blocks = []
    for c in range(nrow):
        for n in range(ncol):
            b = [
                [c_sum[n] - delc[n], r_sum[c] - delr[c]],
                [c_sum[n] - delc[n], r_sum[c]],
                [c_sum[n], r_sum[c]],
                [c_sum[n], r_sum[c] - delr[c]],
            ]
            blocks.append(b)
    blocks = np.array(blocks)

    return blocks


def matrix_paste(c_big, c_small):
    # Compute distance matrix between refined and dummy grid.
    dm = distance_matrix(c_big, c_small)
    inds = [
        np.unravel_index(np.argmin(dm[i], axis=None), dm[i].shape)[0]
        for i in range(dm.shape[0])
    ]
    return inds


def h_sub(h: np.array, un: int, uc: int, sc: int):
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


def get_centroids(array: np.array, grf: float):
    """
    Given a (m, n) matrix of cells dimensions in the x-y axes, returns the (m, n, 2) matrix of the coordinates of
    centroids.
    :param array: (m, n) array
    :param grf: float: Cell dimension
    """
    xys = np.dstack((np.flip((np.indices(array.shape) + 1), 0) * grf -
                     grf / 2))  # Getting centroids
    return xys.reshape((array.shape[0] * array.shape[1], 2))


def contours_vertices(x: list,
                      y: list,
                      arrays: np.array,
                      c: float = 0,
                      ignore_: bool = True):
    """
    Extracts contour vertices from a list of matrices.
    :param x:
    :param y:
    :param arrays: list of matrices
    :param c: Contour value
    :param ignore_: Bool value to consider multiple contours or not (see comments)
    :return: vertices array
    """
    if len(arrays.shape) < 3:
        arrays = [arrays]
    # First create figures for each forecast.
    figs = [plt.figure() for _ in range(len(arrays))]
    c0s = [plt.contour(x, y, f, [c]) for f in arrays]
    [plt.close(f) for f in figs]  # Close plots
    # .allseg[0][0] extracts the vertices of each O contour = WHPA's vertices

    # /!\ If more than one contours are present for the same values, possibility to include them or not.
    # How are the contours sorted in c0.allsegs[0][i] ?
    # It looks like they are sorted by size.
    if ignore_:
        v = np.array([c0.allsegs[0][0] for c0 in c0s], dtype=object)
    else:
        v = np.array(
            [
                c0.allsegs[0][i] for c0 in c0s
                for i in range(len(c0.allsegs[0]))
            ],
            dtype=object,
        )
    return v


def binary_polygon(
        xys: np.array,
        nrow: int,
        ncol: int,
        pzs: np.array,
        outside: float = -1,
        inside: float = 1,
):
    """
    Given a polygon whose vertices are given by the array pzs, and a matrix of
    centroids coordinates of the surface discretization, assigns to the matrix a certain value
    whether the cell is inside or outside said polygon.

    To compute the signed distance function, we need a negative/positive value.

    :param xys: Centroids of a grid' cells
    :param nrow: Number of rows
    :param ncol: Number of columns
    :param pzs: Array of ordered vertices coordinates of a polygon.
    :param pzs: Polygon vertices (v, 2)
    :param outside: Value to assign to the matrix outside of the polygon
    :param inside: Value to assign to the matrix inside of the polygon
    :return: phi = the binary matrix
    """

    # Creates a Polygon abject out of the polygon vertices in pzs
    poly = Polygon(pzs, True)
    # Checks which points are enclosed by polygon.
    ind = np.nonzero(poly.contains_points(xys))[0]
    phi = np.ones((nrow, ncol)) * outside  # SD - create matrix of 'outside'
    phi = phi.reshape((nrow * ncol))  # Flatten to have same dimension as 'ind'
    phi[ind] = inside  # Points inside the WHPA are assigned a value of 'inside'
    phi = phi.reshape((nrow, ncol))  # Reshape

    return phi


def binary_stack(xys, nrow, ncol, vertices):
    """
    Takes WHPA vertices and 'binarizes' the image (e.g. 1 inside, 0 outside WHPA).
    """
    # Create binary images of WHPA stored in bin_whpa
    bin_whpa = [
        binary_polygon(xys, nrow, ncol, pzs=p, inside=1, outside=-1)
        for p in vertices
    ]
    big_sum = np.sum(bin_whpa, axis=0)  # Stack them
    # Scale from 0 to 1
    big_sum -= np.min(big_sum)
    big_sum /= np.max(big_sum)
    return big_sum


def get_block(pm, i: int):
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


def refine_machine(xlim, ylim, new_grf):
    nrow = int(np.diff(ylim) / new_grf)  # Number of rows
    ncol = int(np.diff(xlim) / new_grf)  # Number of columns
    new_x, new_y = np.meshgrid(np.linspace(xlim[0], xlim[1], ncol),
                               np.linspace(ylim[0], ylim[1], nrow))
    return nrow, ncol, new_x, new_y
