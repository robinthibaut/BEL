#  Copyright (c) 2021. Robin Thibaut, Ghent University

import inspect
import warnings
from collections import defaultdict

__all__ = ["BaseEstimator", "RegressorMixin", "TransformerMixin"]


class BaseEstimator:
    """Base class for all estimators in scikit-learn

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]

        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                warnings.warn(
                    "Attribute Error",
                    FutureWarning,
                )
                value = None
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__.copy()

        return state

    def __setstate__(self, state):
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)


class RegressorMixin:
    """Mixin class for all regression estimators in scikit-learn."""

    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a
            precomputed kernel matrix or a list of generic objects instead,
            shape = (n_samples, n_samples_fitted),
            where n_samples_fitted is the number of
            samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.

        """

        from bel4ed.algorithms.metrics import _check_reg_targets, r2_score

        y_pred = self.predict(X)
        # XXX: Remove the check in 0.23
        y_type, _, _, _ = _check_reg_targets(y, y_pred, None)
        return r2_score(
            y, y_pred, sample_weight=sample_weight, multioutput="variance_weighted"
        )


class TransformerMixin:
    """Mixin class for all transformers in scikit-learn."""

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then fit_transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).fit_transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).fit_transform(X)
