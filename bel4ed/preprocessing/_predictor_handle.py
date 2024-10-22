#  Copyright (c) 2021. Robin Thibaut, Ghent University

import numpy as np
from scipy.interpolate import interp1d

__all__ = ["curve_interpolation", "beautiful_curves"]

from bel4ed.datasets import data_loader
from bel4ed.utils import Root


def curve_interpolation(tc0, n_time_steps: int = 200, t_max: float = 1.01080e02):
    """

    Perform data transformations on the predictor.

    The breakthrough curves do not share the same time steps.

    We need to save the data array in a consistent shape, thus interpolates and sub-divides each simulation
    curves into n time steps.

    :param tc0: original data - breakthrough curves of shape (n_sim, n_time_steps, n_wells)
    :param n_time_steps: float: desired number of time step, will be the new dimension in shape[1].
    :param t_max: float: Time corresponding to the end of the simulation (default unit is seconds).
    :return: Observation data array with shape (n_sim, n_time_steps, n_wells)
    """
    # Preprocess d
    f1d = []  # List of interpolating functions for each curve
    for t in tc0:
        fs = [interp1d(c[:, 0], c[:, 1], fill_value="extrapolate") for c in t]
        f1d.append(fs)
    f1d = np.array(f1d)
    # Watch out as the two following variables are also defined in the load_data() function:
    # n_time_steps = 200  Arbitrary number of time steps to create the final transport array
    ls = np.linspace(0, t_max, num=n_time_steps)
    tc = []  # List of interpolating functions for each curve
    for f in f1d:
        ts = [fi(ls) for fi in f]
        tc.append(ts)
    tc = np.array(tc)  # Data array

    return tc


def beautiful_curves(res_dir: str, ids: Root, n_time_steps: int) -> np.array:
    """Loads and process predictor (tracer curves)"""

    tc_training_raw, *_ = data_loader(res_dir=res_dir, roots=ids, d=True)
    tc_training = curve_interpolation(tc0=tc_training_raw, n_time_steps=n_time_steps)

    return tc_training
