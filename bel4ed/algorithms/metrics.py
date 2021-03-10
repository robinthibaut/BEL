"""Metrics to assess performance on regression task

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""
import warnings
from os.path import join

import numpy
import numpy as np
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity
from sklearn.neighbors import KernelDensity

from bel4ed.spatial import grid_parameters, binary_polygon
from bel4ed.utils import (
    _num_samples,
    check_array,
    check_consistent_length,
    column_or_1d,
)

from bel4ed.exceptions import UndefinedMetricWarning

__all__ = ["r2_score", "modified_hausdorff", "structural_similarity"]


def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric"):
    """Check that y_true and y_pred belong to the same regression task

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like

    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().

    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'

    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples, n_outputs)
        Estimated target values.

    multioutput : array-like of shape (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.
    dtype: str or list, default="numeric"
        the dtype argument passed to check_array

    """
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError(
            "y_true and y_pred have different number of output "
            "({0}!={1})".format(y_true.shape[1], y_pred.shape[1])
        )

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ("raw_values", "uniform_average", "variance_weighted")
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError(
                "Allowed 'multioutput' string values are {}. "
                "You provided multioutput={!r}".format(
                    allowed_multioutput_str, multioutput
                )
            )
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in " "multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(
                ("There must be equally many custom weights " "(%d) as outputs (%d).")
                % (len(multioutput), n_outputs)
            )
    y_type = "continuous" if n_outputs == 1 else "continuous-multioutput"

    return y_type, y_true, y_pred, multioutput


def r2_score(y_true, y_pred, sample_weight=None, multioutput="uniform_average"):
    """R^2 (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). A constant model that always
    predicts the expected value of y, disregarding the input features,
    would get a R^2 score of 0.0.

    Read more in the :ref:`User Guide <r2_score>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.

    multioutput : string in ['raw_values', 'uniform_average', \
'variance_weighted'] or None or array-like of shape (n_outputs)

        Defines aggregating of multiple output scores.
        Array-like value defines weights used to average scores.
        Default is "uniform_average".

        'raw_values' :
            Returns a full set of scores in case of multioutput input.

        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.

        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.

            Default value of multioutput is 'uniform_average'.

    Returns
    -------
    z : float or ndarray of floats
        The R^2 score or ndarray of scores if 'multioutput' is
        'raw_values'.

    Notes
    -----
    This is not a symmetric function.

    Unlike most other scores, R^2 score may be negative (it need not actually
    be the square of a quantity R).

    This metric is not well-defined for single samples and will return a NaN
    value if n_samples is less than two.

    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_

    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    check_consistent_length(y_true, y_pred, sample_weight)

    if _num_samples(y_pred) < 2:
        msg = "R^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float("nan")

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.0

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = (
        weight * (y_true - np.average(y_true, axis=0, weights=sample_weight)) ** 2
    ).sum(axis=0, dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            # return scores individually
            return output_scores
        elif multioutput == "uniform_average":
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == "variance_weighted":
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0
    else:
        avg_weights = multioutput

    return np.average(output_scores, weights=avg_weights)


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
    # TODO: create a function to copy/paste values on differently refined grids
    # Prepare the Plot instance with right dimensions
    grf_kd = 4
    cell_dim = grf_kd
    xgrid = np.arange(xmin, xmax, cell_dim)
    ygrid = np.arange(ymin, ymax, cell_dim)
    X, Y = np.meshgrid(xgrid, ygrid)
    # x, y coordinates of the grid cells vertices
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    # Define a disk within which the KDE will be performed to save time
    # TODO: Move this to parameter file
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
    np.save(jp(res_dir, "bin"), b_low)
