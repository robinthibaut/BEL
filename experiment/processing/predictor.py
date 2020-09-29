#  Copyright (c) 2020. Robin Thibaut, Ghent University

import numpy as np
from scipy.interpolate import interp1d


def d_process(tc0,
              n_time_steps: int = 200,
              t_max: float = 1.01080e+02):
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
        fs = [interp1d(c[:, 0], c[:, 1], fill_value='extrapolate') for c in t]
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
