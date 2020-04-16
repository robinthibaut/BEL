from os.path import join as jp
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import PowerTransformer

from bel.toolbox.mesh_ops import MeshOps


class DataOps:

    def __init__(self):
        self.mo = MeshOps()
        self.gaussian_transformers = {}

    @staticmethod
    def d_process(tc0, n_time_steps=500, t_max=1.01080e+02):
        """
        The breakthrough curves do not share the same time steps.
        We need to save the data array in a consistent shape, thus interpolates and sub-divides each simulation
        curves into n time steps.
        @param tc0: original data - breakthrough curves of shape (n_sim, n_time_steps, n_wells)
        @param n_time_steps: desired number of time step, will be the new dimension in shape[1].
        @param t_max: Time corresponding to the end of the simulation.
        @return: Data array with shape (n_sim, n_time_steps, n_wells)
        """
        # Preprocess d
        f1d = []  # List of interpolating functions for each curve
        for t in tc0:
            fs = [interp1d(c[:, 0], c[:, 1], fill_value='extrapolate') for c in t]
            f1d.append(fs)
        f1d = np.array(f1d)
        # Watch out as the two following variables are also defined in the load_data() function:
        # n_time_steps = 500  Arbitrary number of time steps to create the final transport array
        ls = np.linspace(0, t_max, num=n_time_steps)  # From 0 to 200 days with 1000 steps
        tc = []  # List of interpolating functions for each curve
        for f in f1d:
            ts = [fi(ls) for fi in f]
            tc.append(ts)
        tc = np.array(tc)  # Data array

        return tc

    def h_process(self, h, sc=5, wdir=''):
        """
        Process signed distance array.
        @param h: Signed distance array
        @param sc: New cell dimension in x and y direction (original is 1)
        @param wdir: Folder to save the new array into
        """
        nrow, ncol = self.mo.nrow, self.mo.ncol  # Get the info from the MeshOps class
        un, uc = int(nrow / sc), int(ncol / sc)
        h_u = h_sub(h=h, un=un, uc=uc, sc=sc)
        np.save(jp(wdir, 'h_u.npy'), h_u)  # Save transformed function matrix

    def gaussian_distribution(self, original_array, name='gd'):
        # Ensure Gaussian distribution in original_array Each vector for each original_array components will be
        # transformed one-by-one by a different operator, stored in yj.
        yj = [PowerTransformer(method='yeo-johnson', standardize=True) for c in range(original_array.shape[0])]
        self.gaussian_transformers[name] = yj  # Adds the gaussian distribution transformers object to the dictionary
        # Fit each PowerTransformer with each component
        [yj[i].fit(original_array[i].reshape(-1, 1)) for i in range(len(yj))]
        # Transform the original distribution.
        original_array_gaussian \
            = np.concatenate([yj[i].transform(original_array[i].reshape(-1, 1)) for i in range(len(yj))], axis=1).T

        return original_array_gaussian

    def gaussian_inverse(self, original_array, name='gd'):
        yj = self.gaussian_transformers[name]
        back2 \
            = np.concatenate([yj[i].inverse_transform(original_array[i].reshape(-1, 1)) for i in range(len(yj))],
                             axis=1).T

        return back2
