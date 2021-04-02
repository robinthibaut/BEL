#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os

import numpy as np
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity
from sklearn.neighbors import KernelDensity

from bel4ed.spatial import grid_parameters, binary_polygon

__all__ = ["modified_hausdorff", "structural_similarity"]


def modified_hausdorff(a: np.array, b: np.array) -> float:
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


def kernel_density(x_lim, y_lim, vertices):
    # Scatter plot vertices
    # nn = sample_n
    # plt.plot(vertices[nn][:, 0], vertices[nn][:, 1], 'o-')
    # plt.show()

    # Grid geometry
    xmin = x_lim[0]
    xmax = x_lim[1]
    ymin = y_lim[0]
    ymax = y_lim[1]
    # Create a structured grid to estimate kernel density
    # Prepare the Plot instance with right dimensions
    grf_kd = 4
    cell_dim = grf_kd
    xgrid = np.arange(xmin, xmax, cell_dim)
    ygrid = np.arange(ymin, ymax, cell_dim)
    X, Y = np.meshgrid(xgrid, ygrid)
    # x, y coordinates of the grid cells vertices
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    # Define a disk within which the KDE will be performed to save time
    x0, y0, radius = 1000, 500, 200
    r = np.sqrt((xy[:, 0] - x0) ** 2 + (xy[:, 1] - y0) ** 2)
    inside = r < radius
    xyu = xy[inside]  # Create mask

    # Perform KDE
    bw = 1.0  # Arbitrary 'smoothing' parameter
    # Reshape coordinates
    x_stack = np.hstack([vi[:, 0] for vi in vertices])
    y_stack = np.hstack([vi[:, 1] for vi in vertices])
    # Final array np.array([[x0, y0],...[xn,yn]])
    xykde = np.vstack([x_stack, y_stack]).T
    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(  # Fit kernel density
        xykde
    )
    # Sample at the desired grid cells
    score = np.exp(kde.score_samples(xyu))

    def score_norm(sc, max_score=None):
        """
        Normalizes the KDE scores.
        """
        sc -= sc.min()
        sc /= sc.max()

        sc += 1
        sc = sc ** -1

        sc -= sc.min()
        sc /= sc.max()

        return sc

    # Normalize
    score = score_norm(score)

    # Assign the computed scores to the grid
    z = np.full(inside.shape, 1, dtype=float)  # Create array filled with 1
    z[inside] = score
    # Flip to correspond to actual distribution.
    z = np.flipud(z.reshape(X.shape))

    return z


def uq_binary_stack(x_lim, y_lim, grf, vertices, n_posts, res_dir):
    """
    Takes WHPA vertices and binarizes the image (e.g. 1 inside, 0 outside WHPA).
    """
    xys, nrow, ncol = grid_parameters(
        x_lim=x_lim, y_lim=y_lim, grf=grf
    )  # Initiate SD object
    # Create binary images of WHPA stored in bin_whpa
    bin_whpa = [
        binary_polygon(xys, nrow, ncol, pzs=p, inside=1 / n_posts, outside=0)
        for p in vertices
    ]
    big_sum = np.sum(bin_whpa, axis=0)  # Stack them
    b_low = np.where(big_sum == 0, 1, big_sum)  # Replace 0 values by 1
    b_low = np.flipud(b_low)

    # Save result
    np.save(os.path.join(res_dir, "bin"), b_low)
