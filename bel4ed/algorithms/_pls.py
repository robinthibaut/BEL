from abc import ABCMeta, abstractmethod

import numpy as np
from loguru import logger
from scipy.linalg import pinv2, svd

from bel4ed.utils import (
    FLOAT_DTYPES,
    check_array,
    check_consistent_length,
    check_is_fitted,
)

from ._base import BaseEstimator, RegressorMixin, TransformerMixin
from .extmath import svd_flip

__all__ = ["_PLS"]


def _nipals_twoblocks_inner_loop(X, Y, max_iter=500, tol=1e-06, norm_y_weights=False):
    """Inner loop of the iterative NIPALS algorithm.

    Provides an alternative to the svd(X'Y); returns the first left and right
    singular vectors of X'Y.  See PLS for the meaning of the parameters.  It is
    similar to the Power method for determining the eigenvectors and
    eigenvalues of a X'Y.
    """
    for col in Y.T:
        if np.any(np.abs(col) > np.finfo(np.double).eps):
            y_score = col.reshape(len(col), 1)
            break

    x_weights_old = 0
    ite = 1
    X_pinv = Y_pinv = None
    eps = np.finfo(X.dtype).eps

    # Uses condition from scipy<1.3 in pinv2 which was changed in
    # https://github.com/scipy/scipy/pull/10067. In scipy 1.3, the
    # condition was changed to depend on the largest singular value
    X_t = X.dtype.char.lower()
    Y_t = Y.dtype.char.lower()
    factor = {"f": 1e3, "d": 1e6}

    cond_X = factor[X_t] * eps
    cond_Y = factor[Y_t] * eps

    # Inner loop of the Wold algo.
    while True:

        # We use slower pinv2 (same as np.linalg.pinv) for stability
        # reasons
        X_pinv = pinv2(X, check_finite=False, cond=cond_X)
        x_weights = np.dot(X_pinv, y_score)

        if np.dot(x_weights.T, x_weights) < eps:
            x_weights += eps
        # 1.2 Normalize u
        x_weights /= np.sqrt(np.dot(x_weights.T, x_weights)) + eps
        # 1.3 Update x_score: the X latent scores
        x_score = np.dot(X, x_weights)
        # 2.1 Update y_weights

        if Y_pinv is None:
            # compute once pinv(Y)
            Y_pinv = pinv2(Y, check_finite=False, cond=cond_Y)
        y_weights = np.dot(Y_pinv, x_score)

        # 2.2 Normalize y_weights
        if norm_y_weights:
            y_weights /= np.sqrt(np.dot(y_weights.T, y_weights)) + eps
        # 2.3 Update y_score: the Y latent scores
        y_score = np.dot(Y, y_weights) / (np.dot(y_weights.T, y_weights) + eps)
        # y_score = np.dot(Y, y_weights) / np.dot(y_score.T, y_score) ## BUG
        x_weights_diff = x_weights - x_weights_old
        if np.dot(x_weights_diff.T, x_weights_diff) < tol or Y.shape[1] == 1:
            break
        if ite == max_iter:
            msg = "Maximum number of iterations reached"
            logger.warning(msg)
            break
        x_weights_old = x_weights
        ite += 1
    return x_weights, y_weights, ite


def _svd_cross_product(X, Y):
    C = np.dot(X.T, Y)
    U, s, Vh = svd(C, full_matrices=False, compute_uv=True)
    u = U[:, [0]]
    v = Vh.T[:, [0]]
    return u, v


def _center_scale_xy(X, Y, scale=True):
    """Center X, Y and scale if the scale parameter==True

    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    # center
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = Y.mean(axis=0)
    Y -= y_mean
    # scale
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])
    return X, Y, x_mean, y_mean, x_std, y_std


class _PLS(TransformerMixin, RegressorMixin, BaseEstimator, metaclass=ABCMeta):
    """Partial Least Squares (PLS)

    This class implements the generic PLS algorithm, constructors' parameters
    allow to obtain a specific implementation such as:

    We use the terminology defined by [Wegelin et al. 2000].
    This implementation uses the PLS Wold 2 blocks algorithm based on two
    nested loops:
        (i) The outer loop iterate over components.
        (ii) The inner loop estimates the weights vectors. This can be done
        with one algo: the inner loop of the original NIPALS algo.

    n_components : int, number of components to keep. (default 2).

    scale : boolean, scale data? (default True)

    deflation_mode : str, "canonical" or "regression". See notes.

    norm_y_weights : boolean, normalize Y weights to one? (default False)

    algorithm : string, "nipals"
        The algorithm used to estimate the weights. It will be called
        n_components times, i.e. once for each iteration of the outer loop.

    max_iter : int (default 500)
        The maximum number of iterations
        of the NIPALS inner loop

    tol : non-negative real, default 1e-06
        The tolerance used in the iterative algorithm.

    copy : boolean, default True
        Whether the deflation should be done on a copy. Let the default
        value to True unless you don't care about side effects.

    Attributes
    ----------
    x_weights_ : array, [p, n_components]
        X block weights vectors.

    y_weights_ : array, [q, n_components]
        Y block weights vectors.

    x_loadings_ : array, [p, n_components]
        X block loadings vectors.

    y_loadings_ : array, [q, n_components]
        Y block loadings vectors.

    x_scores_ : array, [n_samples, n_components]
        X scores.

    y_scores_ : array, [n_samples, n_components]
        Y scores.

    x_rotations_ : array, [p, n_components]
        X block to latents rotations.

    y_rotations_ : array, [q, n_components]
        Y block to latents rotations.

    x_mean_ : array, [p]
        X mean for each predictor.

    y_mean_ : array, [q]
        Y mean for each response variable.

    x_std_ : array, [p]
        X standard deviation for each predictor.

    y_std_ : array, [q]
        Y standard deviation for each response variable.

    coef_ : array, [p, q]
        The coefficients of the linear model: ``Y = X coef_ + Err``

    n_iter_ : array-like
        Number of iterations of the NIPALS inner loop for each
        component. Not useful if the algorithm given is "svd".

    References
    ----------

    Jacob A. Wegelin. A survey of Partial Least Squares (PLS) methods, with
    emphasis on the two-block case. Technical Report 371, Department of
    Statistics, University of Washington, Seattle, 2000.

    In French but still a reference:
    Tenenhaus, M. (1998). La regression PLS: theorie et pratique. Paris:
    Editions Technic.

    """

    @abstractmethod
    def __init__(
        self,
        n_components=2,
        scale=True,
        deflation_mode="regression",
        norm_y_weights=False,
        max_iter=500,
        tol=1e-06,
        copy=True,
    ):
        self.n_components = n_components
        self.deflation_mode = deflation_mode
        self.norm_y_weights = norm_y_weights
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    def fit(self, X, Y):
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
        """

        # copy since this will contains the residuals (deflated) matrices
        check_consistent_length(X, Y)
        X = check_array(X, dtype=np.float64, copy=self.copy, ensure_min_samples=2)
        Y = check_array(Y, dtype=np.float64, copy=self.copy, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]

        if self.n_components < 1 or self.n_components > p:
            raise ValueError("Invalid number of components: %d" % self.n_components)
        if self.deflation_mode not in ["canonical", "regression"]:
            raise ValueError("The deflation mode is unknown")
        # Scale (in place)
        X, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = _center_scale_xy(
            X, Y, self.scale
        )
        # Residuals (deflated) matrices
        Xk = X
        Yk = Y
        # Results matrices
        self.x_scores_ = np.zeros((n, self.n_components))
        self.y_scores_ = np.zeros((n, self.n_components))
        self.x_weights_ = np.zeros((p, self.n_components))
        self.y_weights_ = np.zeros((q, self.n_components))
        self.x_loadings_ = np.zeros((p, self.n_components))
        self.y_loadings_ = np.zeros((q, self.n_components))
        self.n_iter_ = []

        # NIPALS algo: outer loop, over components
        Y_eps = np.finfo(Yk.dtype).eps
        for k in range(self.n_components):
            if np.all(np.dot(Yk.T, Yk) < np.finfo(np.double).eps):
                # Yk constant
                logger.warning("Y residual constant at iteration %s" % k)
                break
            # 1) weights estimation (inner loop)
            # -----------------------------------
            # Replace columns that are all close to zero with zeros
            Yk_mask = np.all(np.abs(Yk) < 10 * Y_eps, axis=0)
            Yk[:, Yk_mask] = 0.0

            x_weights, y_weights, n_iter_ = _nipals_twoblocks_inner_loop(
                X=Xk,
                Y=Yk,
                max_iter=self.max_iter,
                tol=self.tol,
                norm_y_weights=self.norm_y_weights,
            )
            self.n_iter_.append(n_iter_)
            # Forces sign stability of x_weights and y_weights
            # from platform dependent computation if algorithm == 'nipals'
            x_weights, y_weights = svd_flip(x_weights, y_weights.T)
            y_weights = y_weights.T
            # compute scores
            x_scores = np.dot(Xk, x_weights)
            if self.norm_y_weights:
                y_ss = 1
            else:
                y_ss = np.dot(y_weights.T, y_weights)
            y_scores = np.dot(Yk, y_weights) / y_ss
            # test for null variance
            if np.dot(x_scores.T, x_scores) < np.finfo(np.double).eps:
                logger.warning("X scores are null at iteration %s" % k)
                break
            # 2) Deflation (in place)
            # ----------------------
            # Possible memory footprint reduction may done here: in order to
            # avoid the allocation of a data chunk for the rank-one
            # approximations matrix which is then subtracted to Xk, we suggest
            # to perform a column-wise deflation.
            #
            # - regress Xk's on x_score
            x_loadings = np.dot(Xk.T, x_scores) / np.dot(x_scores.T, x_scores)
            # - subtract rank-one approximations to obtain remainder matrix
            Xk -= np.dot(x_scores, x_loadings.T)
            if self.deflation_mode == "canonical":
                # - regress Yk's on y_score, then subtract rank-one approx.
                y_loadings = np.dot(Yk.T, y_scores) / np.dot(y_scores.T, y_scores)
                Yk -= np.dot(y_scores, y_loadings.T)
            if self.deflation_mode == "regression":
                # - regress Yk's on x_score, then subtract rank-one approx.
                y_loadings = np.dot(Yk.T, x_scores) / np.dot(x_scores.T, x_scores)
                Yk -= np.dot(x_scores, y_loadings.T)
            # 3) Store weights, scores and loadings # Notation:
            self.x_scores_[:, k] = x_scores.ravel()  # T
            self.y_scores_[:, k] = y_scores.ravel()  # U
            self.x_weights_[:, k] = x_weights.ravel()  # W
            self.y_weights_[:, k] = y_weights.ravel()  # C
            self.x_loadings_[:, k] = x_loadings.ravel()  # P
            self.y_loadings_[:, k] = y_loadings.ravel()  # Q
        # Such that: X = TP' + Err and Y = UQ' + Err

        # 4) rotations from input space to transformed space (scores)
        # T = X W(P'W)^-1 = XW* (W* : p x k matrix)
        # U = Y C(Q'C)^-1 = YC* (W* : q x k matrix)
        self.x_rotations_ = np.dot(
            self.x_weights_,
            pinv2(np.dot(self.x_loadings_.T, self.x_weights_), check_finite=False),
        )
        if Y.shape[1] > 1:
            self.y_rotations_ = np.dot(
                self.y_weights_,
                pinv2(np.dot(self.y_loadings_.T, self.y_weights_), check_finite=False),
            )
        else:
            self.y_rotations_ = np.ones(1)

        # Estimate regression coefficient
        # Regress Y on T
        # Y = TQ' + Err,
        # Then express in function of X
        # Y = X W(P'W)^-1Q' + Err = XB + Err
        # => B = W*Q' (p x q)
        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
        self.coef_ = self.coef_ * self.y_std_

        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.

        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        check_is_fitted(self)
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_
        # Apply rotation
        x_scores = np.dot(X, self.x_rotations_)
        if Y is not None:
            Y = check_array(Y, ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Y -= self.y_mean_
            Y /= self.y_std_
            y_scores = np.dot(Y, self.y_rotations_)
            return x_scores, y_scores

        return x_scores

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of pls components.

        Returns
        -------
        x_reconstructed : array-like of shape (n_samples, n_features)

        Notes
        -----
        This transformation will only be exact if n_components=n_features
        """
        check_is_fitted(self)
        X = check_array(X, dtype=FLOAT_DTYPES)
        # From pls space to original space
        X_reconstructed = np.matmul(X, self.x_loadings_.T)

        # Denormalize
        X_reconstructed *= self.x_std_
        X_reconstructed += self.x_mean_
        return X_reconstructed

    def predict(self, X, copy=True):
        """Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Notes
        -----
        This call requires the estimation of a p x q matrix, which may
        be an issue in high dimensional space.
        """
        check_is_fitted(self)
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_
        Ypred = np.dot(X, self.coef_)
        return Ypred + self.y_mean_

    def fit_transform(self, X, y=None):
        """Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        return self.fit(X, y).transform(X, y)
