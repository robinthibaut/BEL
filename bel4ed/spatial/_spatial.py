#  Copyright (c) 2021. Robin Thibaut, Ghent University
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams.update({"figure.max_open_warning": 0})

__all__ = [
    "binary_polygon",
    "binary_stack",
]


def binary_polygon(
    xys: np.array,
    nrow: int,
    ncol: int,
    pzs: np.array,
    outside: float = -1,
    inside: float = 1,
) -> np.array:
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


def binary_stack(xys: np.array, nrow: int, ncol: int, vertices: np.array) -> np.array:
    """
    Takes WHPA vertices and 'binarizes' the image (e.g. 1 inside, 0 outside WHPA).
    """
    # Create binary images of WHPA stored in bin_whpa
    bin_whpa = [
        binary_polygon(xys, nrow, ncol, pzs=p, inside=1, outside=-1) for p in vertices
    ]
    big_sum = np.sum(bin_whpa, axis=0)  # Stack them
    # Scale from 0 to 1
    big_sum -= np.min(big_sum)
    big_sum /= np.max(big_sum)
    return big_sum
