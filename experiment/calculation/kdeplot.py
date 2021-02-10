#  Copyright (c) 2021. Robin Thibaut, Ghent University
import datetime
import itertools
import math
import warnings
from numbers import Number

import pandas as pd
import numpy as np
from seaborn._core import (
    VectorPlotter,
)
from seaborn._statistics import (
    KDE,
)


def variable_type(vector, boolean_type="numeric"):
    """
    Determine whether a vector contains numeric, categorical, or datetime data.

    This function differs from the pandas typing API in two ways:

    - Python sequences or object-typed PyData objects are considered numeric if
      all of their entries are numeric.
    - String or mixed-type data are considered categorical even if not
      explicitly represented as a :class:`pandas.api.types.CategoricalDtype`.

    Parameters
    ----------
    vector : :func:`pandas.Series`, :func:`numpy.ndarray`, or Python sequence
        Input data to test.
    boolean_type : 'numeric' or 'categorical'
        Type to use for vectors containing only 0s and 1s (and NAs).

    Returns
    -------
    var_type : 'numeric', 'categorical', or 'datetime'
        Name identifying the type of data in the vector.
    """
    # If a categorical dtype is set, infer categorical
    if pd.api.types.is_categorical_dtype(vector):
        return "categorical"

    # Special-case all-na data, which is always "numeric"
    if pd.isna(vector).all():
        return "numeric"

    # Special-case binary/boolean data, allow caller to determine
    # This triggers a numpy warning when vector has strings/objects
    # https://github.com/numpy/numpy/issues/6784
    # Because we reduce with .all(), we are agnostic about whether the
    # comparison returns a scalar or vector, so we will ignore the warning.
    # It triggers a separate DeprecationWarning when the vector has datetimes:
    # https://github.com/numpy/numpy/issues/13548
    # This is considered a bug by numpy and will likely go away.
    with warnings.catch_warnings():
        warnings.simplefilter(
            action='ignore', category=(FutureWarning, DeprecationWarning)
        )
        if np.isin(vector, [0, 1, np.nan]).all():
            return boolean_type

    # Defer to positive pandas tests
    if pd.api.types.is_numeric_dtype(vector):
        return "numeric"

    if pd.api.types.is_datetime64_dtype(vector):
        return "datetime"

    # --- If we get to here, we need to check the entries

    # Check for a collection where everything is a number

    def all_numeric(x):
        for x_i in x:
            if not isinstance(x_i, Number):
                return False
        return True

    if all_numeric(vector):
        return "numeric"

    # Check for a collection where everything is a datetime

    def all_datetime(x):
        for x_i in x:
            if not isinstance(x_i, (datetime, np.datetime64)):
                return False
        return True

    if all_datetime(vector):
        return "datetime"

    # Otherwise, our final fallback is to consider things categorical

    return "categorical"


class _DistributionPlotter():
    semantics = "x", "y", "hue", "weights"

    wide_structure = {"x": "@values", "hue": "@columns"}
    flat_structure = {"x": "@values"}

    def __init__(self, data=None, variables={}):

        self.assign_variables(data, variables)

        self._var_levels = {}

    def assign_variables(self, data=None, variables={}):
        """Define plot variables, optionally using lookup from `data`."""
        x = variables.get("x", None)
        y = variables.get("y", None)

        if x is None and y is None:
            self.input_format = "wide"
            plot_data, variables = self._assign_variables_wideform(
                data, **variables,
            )
        else:
            self.input_format = "long"
            plot_data, variables = self._assign_variables_longform(
                data, **variables,
            )

        self.plot_data = plot_data
        self.variables = variables
        self.var_types = {
            v: variable_type(
                plot_data[v],
                boolean_type="numeric" if v in "xy" else "categorical"
            )
            for v in variables
        }

        return self

    def iter_data(
            self, grouping_vars=None, reverse=False, from_comp_data=False,
    ):
        """Generator for getting subsets of data defined by semantic variables.

        Also injects "col" and "row" into grouping semantics.

        Parameters
        ----------
        grouping_vars : string or list of strings
            Semantic variables that define the subsets of data.
        reverse : bool, optional
            If True, reverse the order of iteration.
        from_comp_data : bool, optional
            If True, use self.comp_data rather than self.plot_data

        Yields
        ------
        sub_vars : dict
            Keys are semantic names, values are the level of that semantic.
        sub_data : :class:`pandas.DataFrame`
            Subset of ``plot_data`` for this combination of semantic values.

        """
        # TODO should this default to using all (non x/y?) semantics?
        # or define groupping vars somewhere?
        if grouping_vars is None:
            grouping_vars = []
        elif isinstance(grouping_vars, str):
            grouping_vars = [grouping_vars]
        elif isinstance(grouping_vars, tuple):
            grouping_vars = list(grouping_vars)

        # Always insert faceting variables
        facet_vars = {"col", "row"}
        grouping_vars.extend(
            facet_vars & set(self.variables) - set(grouping_vars)
        )

        # Reduce to the semantics used in this plot
        grouping_vars = [
            var for var in grouping_vars if var in self.variables
        ]

        if from_comp_data:
            data = self.comp_data
        else:
            data = self.plot_data

        if grouping_vars:

            grouped_data = data.groupby(
                grouping_vars, sort=False, as_index=False
            )

            grouping_keys = []
            for var in grouping_vars:
                grouping_keys.append(self.var_levels.get(var, []))

            iter_keys = itertools.product(*grouping_keys)
            if reverse:
                iter_keys = reversed(list(iter_keys))

            for key in iter_keys:

                # Pandas fails with singleton tuple inputs
                pd_key = key[0] if len(key) == 1 else key

                try:
                    data_subset = grouped_data.get_group(pd_key)
                except KeyError:
                    continue

                sub_vars = dict(zip(grouping_vars, key))

                yield sub_vars, data_subset

        else:

            yield {}, data

    @property
    def has_xy_data(self):
        """Return True at least one of x or y is defined."""
        # TODO see above points about where this should go
        return bool({"x", "y"} & set(self.variables))

    def _quantile_to_level(self, data, quantile):
        """Return data levels corresponding to quantile cuts of mass."""
        isoprop = np.asarray(quantile)
        values = np.ravel(data)
        sorted_values = np.sort(values)[::-1]
        normalized_values = np.cumsum(sorted_values) / values.sum()
        idx = np.searchsorted(normalized_values, 1 - isoprop)
        levels = np.take(sorted_values, idx, mode="clip")
        return levels

    # -------------------------------------------------------------------------------- #
    # Computation
    # -------------------------------------------------------------------------------- #

    def plot_bivariate_density(
            self,
            common_norm,
            estimate_kws,
    ):

        estimator = KDE(**estimate_kws)

        if not set(self.variables) - {"x", "y"}:
            common_norm = False

        all_data = self.plot_data.dropna()

        # Loop through the subsets and estimate the KDEs
        densities, supports = {}, {}

        for sub_vars, sub_data in self.iter_data("hue", from_comp_data=True):

            # Extract the data points from this sub set and remove nulls
            sub_data = sub_data.dropna()
            observations = sub_data[["x", "y"]]

            # Extract the weights for this subset of observations
            if "weights" in self.variables:
                weights = sub_data["weights"]
            else:
                weights = None

            # Check that KDE will not error out
            variance = observations[["x", "y"]].var()
            if any(math.isclose(x, 0) for x in variance) or variance.isna().any():
                msg = "Dataset has 0 variance; skipping density estimate."
                warnings.warn(msg, UserWarning)
                continue

            # Estimate the density of observations at this level
            observations = observations["x"], observations["y"]
            density, support = estimator(*observations, weights=weights)

            # Transform the support grid back to the original scale
            xx, yy = support
            if self._log_scaled("x"):
                xx = np.power(10, xx)
            if self._log_scaled("y"):
                yy = np.power(10, yy)
            support = xx, yy

            # Apply a scaling factor so that the integral over all subsets is 1
            if common_norm:
                density *= len(sub_data) / len(all_data)

            key = tuple(sub_vars.items())
            densities[key] = density
            supports[key] = support


def kdeplot(
        x=None,  # Allow positional x, because behavior will not change with reorg
        *,
        y=None,
        kernel=None,  # Deprecated
        bw=None,  # Deprecated
        gridsize=200,
        cut=3, clip=None, legend=True, cumulative=False,
        shade_lowest=None,  # Deprecated, controlled with levels now
        cbar=False, cbar_ax=None, cbar_kws=None,
        ax=None,

        # New params
        weights=None,
        hue=None, palette=None, hue_order=None, hue_norm=None,
        multiple="layer", common_norm=True, common_grid=False,
        levels=10, thresh=.05,
        bw_method="scott", bw_adjust=1, log_scale=None,
        color=None, fill=None,

        # Renamed params
        data=None, data2=None,

        **kwargs,
):
    # Handle deprecation of `bw`
    if bw is not None:
        bw_method = bw

    p = _DistributionPlotter(
        data=data,
        variables=_DistributionPlotter.get_semantics(locals()),
    )

    # Pack the kwargs for statistics.KDE
    estimate_kws = dict(
        bw_method=bw_method,
        bw_adjust=bw_adjust,
        gridsize=gridsize,
        cut=cut,
        clip=clip,
        cumulative=cumulative,
    )

    p.plot_bivariate_density(
        common_norm=common_norm,
        levels=levels,
        thresh=thresh,
        estimate_kws=estimate_kws,
        **kwargs,
    )

    return ax
