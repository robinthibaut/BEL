import os
from os.path import join as jp
import shutil

import joblib

import flopy
import matplotlib.pyplot as plt
import numpy as np
from numpy.matlib import repmat
from scipy.interpolate import interp1d

from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer


class FileOps:

    def __init__(self):
        pass

    @staticmethod
    def load_data(res_dir, n=0, check=True, data_flag=False):
        """

        @param check: Whether to check breaktrough curves for unrealistic results or not.
        @param res_dir: main directory containing results sub-directories
        @param n: if != 0, will randomly select n sub-folders from res_dir
        @param data_flag: if True, only loads data and not the target
        @return: tp, sd
        """
        # TODO: Split this function to generalize and load one feature at a time

        bkt_files = []  # Breakthrough curves files
        sd_files = []  # Signed-distance files
        hk_files = []  # Hydraulic conductivity files
        # r=root, d=directories, f = files
        roots = []
        for r, d, f in os.walk(res_dir, topdown=False):
            # Adds the data files to the lists, which will be loaded later
            if 'bkt.npy' in f and 'sd.npy' in f and 'hk0.npy' in f:
                bkt_files.append(jp(r, 'bkt.npy'))
                sd_files.append(jp(r, 'sd.npy'))
                hk_files.append(jp(r, 'hk0.npy'))
                roots.append(r)
            else:  # If one of the files is missing, deletes the sub-folder
                try:
                    if r != res_dir:  # Make sure to not delete the main results directory !
                        shutil.rmtree(r)
                except TypeError:
                    pass
        if n:
            folders = np.random.choice(np.arange(len(roots)), n)  # Randomly selects n folders
            bkt_files = np.array(bkt_files)[folders]
            sd_files = np.array(sd_files)[folders]
            hk_files = np.array(hk_files)[folders]
            roots = np.array(roots)[folders]

        # Load and filter results
        # We first load the breakthrough curves and check for errors.
        # If one of the concentration is > 1, it means the simulation is unrealistic and we remove it from the dataset.
        if check:  # Check whether to delete for simulation issues
            tpt = list(map(np.load, bkt_files))
            rm = []  # Will contain indices to remove
            for i in range(len(tpt)):
                for j in range(len(tpt[i])):
                    if max(tpt[i][j][:, 1]) > 1:  # Check results files whose max computed head is > 1 and removes them
                        rm.append(i)
                        break
            for index in sorted(rm, reverse=True):
                del bkt_files[index]
                del sd_files[index]
                del hk_files[index]
                del roots[index]

        # Check unpacking
        tpt = list(map(np.load, bkt_files))  # Re-load transport curves
        # hk = np.array(list(map(np.load, hk_files)))  # Load hydraulic K
        if not data_flag:
            sd = np.array(list(map(np.load, sd_files)))  # Load signed distance
            return tpt, sd, roots
        else:
            return tpt, roots

    @staticmethod
    def datread(file=None, header=0):
        """Reads space separated dat file"""
        with open(file, 'r') as fr:
            op = np.array([list(map(float, l.split())) for l in fr.readlines()[header:]])
        return op

    @staticmethod
    def dirmaker(dird):
        """
        Given a folder path, check if it exists, and if not, creates it
        :param dird: path
        :return:
        """
        try:
            if not os.path.exists(dird):
                os.makedirs(dird)
        except:
            pass

    @staticmethod
    def load_flow_model(nam_file, exe_name='', model_ws=''):
        flow_loader = flopy.modflow.mf.Modflow.load

        return flow_loader(f=nam_file, exe_name=exe_name, model_ws=model_ws)

    @staticmethod
    def load_transport_model(nam_file, modflowmodel, exe_name='', model_ws='', ftl_file='mt3d_link.ftl',
                             version='mt3d-usgs'):
        transport_loader = flopy.mt3d.Mt3dms.load
        transport_reloaded = transport_loader(f=nam_file, version=version, modflowmodel=modflowmodel,
                                              exe_name=exe_name, model_ws=model_ws)
        transport_reloaded.ftlfilename = ftl_file

        return transport_reloaded


class MeshOps:

    def __init__(self):
        self.xlim = 1500
        self.ylim = 1000
        self.grf = 1  # Cell dimension (1m)
        self.nrow = self.ylim // self.grf
        self.ncol = self.xlim // self.grf

    @staticmethod
    def refine_axis(widths, r_pt, ext, cnd, d_dim, a_lim):
        # TODO: write better documentation for this
        x0 = widths
        x0s = np.cumsum(x0)  # Cumulative sum of the width of the cells
        pt = r_pt
        extx = ext
        cdrx = cnd
        dx = d_dim
        xlim = a_lim

        # X range of the polygon
        xrp = [pt - extx, pt + extx]

        wherex = np.where((xrp[0] < x0s) & (x0s <= xrp[1]))[0]

        # The algorithm must choose a 'flexible parameter', either the cell grid size, the dimensions of the grid or the
        # refined cells themselves
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

            cs = np.cumsum(
                x0)  # Cumulative width should equal xlim, but it will not be the case, have to adapt width
            difx = xlim - cs[-1]
            where_default = np.where(abs(x0 - dx) <= 5)[0]  # Location of cells whose widths will be adapted
            where_left = where_default[
                np.where(where_default < wherex[0])]  # Where do we have the default cell size on the
            # left
            where_right = where_default[np.where((where_default >= wherex[0] + len(nwxs)))]  # And on the right
            lwl = len(where_left)
            lwr = len(where_right)

            if lwl > lwr:
                rl = lwl / lwr  # Weights how many cells are on either sides of the refinement zone
                dal = difx / ((lwl + lwr) / lwl)  # Splitting the extra widths on the left and right of the cells
                dal = dal + (difx - dal) / rl
                dar = difx - dal
            elif lwr > lwl:
                rl = lwr / lwl  # Weights how many cells are on either sides of the refinement zone
                dar = difx / ((lwl + lwr) / lwr)  # Splitting the extra widths on the left and right of the cells
                dar = dar + (difx - dar) / rl
                dal = difx - dar
            else:
                dal = difx / ((lwl + lwr) / lwl)  # Splitting the extra widths on the left and right of the cells
                dar = difx - dal

            x0[where_left] = x0[where_left] + dal / lwl
            x0[where_right] = x0[where_right] + dar / lwr

        return x0  # Flip to correspond to flopy expectations

    @staticmethod
    def blockshaped(arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array should look like n sub-blocks with
        each sub-block preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        assert h % nrows == 0, "{} rows is not evenly divisible by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisible by {}".format(w, ncols)

        return (arr.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

    def h_sub(self, h, un, uc, sc):
        h_u = np.zeros((h.shape[0], un, uc))
        for i in range(h.shape[0]):
            sim = h[i]
            sub = self.blockshaped(sim, sc, sc)
            h_u[i] = np.array([s.mean() for s in sub]).reshape(un, uc)

        return h_u


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
        h_u = self.mo.h_sub(h=h, un=un, uc=uc, sc=sc)
        np.save(jp(wdir, 'h_u.npy'), h_u)  # Save transformed SD matrix

    def gaussian_distribution(self, original_array, name='gd'):
        # Ensure Gaussian distribution in original_array Each vector for each original_array components will be
        # transformed one-by-one by a different operator, stored in yj.
        yj = [PowerTransformer(method='yeo-johnson', standardize=True) for c in range(original_array.shape[0])]
        self.gaussian_transformers[name] = yj  # Adds the gaussian distribution tranformers object to the dictionary
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


class PCAOps:

    def __init__(self, name, raw_data):
        """

        @param name: name of the paramater on which to perform operations
        @param raw_data: original dataset
        """
        self.name = name
        self.raw_data = raw_data  # raw data
        self.n_training = None  # Number of training data
        self.operator = None  # PCA operator
        self.ncomp = None  # Number of components
        self.d0 = None  # Original data
        self.dt = None  # Training set - physical space
        self.dp = None  # Prediction set - physical space
        self.dtp = None  # Training set - PCA space
        self.dpp = None  # Prediction set - PCA space

    def pca_tp(self, n_training):
        """
        Given an arbitrary size of training data, splits the original array accordingly
        @param n_training:
        @return: training, test
        """
        self.n_training = n_training
        # Flattens the array
        d_original = np.array([item for sublist in self.raw_data for item in sublist]).reshape(len(self.raw_data), -1)
        self.d0 = d_original
        # Splits into training and test according to chosen n_training.
        d_t = d_original[:self.n_training]
        self.dt = d_t
        d_p = d_original[self.n_training:]
        self.dp = d_p

        return d_t, d_p

    def pca_transformation(self, load=False):
        """
        Instantiate the PCA object and transforms both training and test data.
        Depending on the value of the load parameter, it will create a new one or load a previously computed one,
        stored in the 'temp' folder.
        @param load:
        @return: PC training, PC test
        """
        if not load:
            pca_operator = PCA()
            self.operator = pca_operator
            pca_operator.fit(self.dt)  # Principal components
            joblib.dump(pca_operator, jp(os.getcwd(), 'temp', '{}_pca_operator.pkl'.format(self.name)))
        else:
            pca_operator = joblib.load(jp(os.getcwd(), 'temp', '{}_pca_operator.pkl'.format(self.name)))
            self.operator = pca_operator

        pc_training = pca_operator.transform(self.dt)  # Principal components
        self.dtp = pc_training
        pc_prediction = pca_operator.transform(self.dp)
        self.dpp = pc_prediction

        return pc_training, pc_prediction

    def n_pca_components(self, perc):
        """
        Given an explained variance percentage, returns the number of components necessary to obtain that level.
        """
        evr = np.cumsum(self.operator.explained_variance_ratio_)
        nc = len(np.where(evr <= perc)[0])

        return nc

    def perc_pca_components(self, n_c):
        """
        Returns the explained variance percentage given a number of components n_c
        """
        evr = np.cumsum(self.operator.explained_variance_ratio_)

        return evr[n_c - 1]

    def pca_refresh(self, n_comp):
        """
        Given a number of components to keep, returns the PC array with the corresponding shape.
        @param n_comp:
        @return:
        """

        self.ncomp = n_comp  # Assign the number of components in the class for later use

        pc_training = self.dtp.copy()  # Reloads the original training components
        pc_training = pc_training[:, :n_comp]  # Cut

        pc_prediction = self.dpp.copy()  # Reloads the original test components
        pc_prediction = pc_prediction[:, :n_comp]  # Cut

        return pc_training, pc_prediction

    def pc_random(self, n_posts):
        """
        Randomly selects PC components from the original training matrix dtp
        """
        r_rows = np.random.choice(self.dtp.shape[0], n_posts)  # Selects n_posts rows from the dtp array
        score_selection = self.dtp[r_rows, self.ncomp:]  # Extracts those rows, from the number of components used until
        # the end of the array.

        # For each column of shape n_sim-ncomp, selects a random PC component to add.
        test = [np.random.choice(score_selection[:, i]) for i in range(score_selection.shape[1])]

        return np.array(test)

    def inverse_transform(self, pc_to_invert):
        """
        Inverse transform PC
        @param pc_to_invert: PC array
        @return:
        """
        inv = np.dot(pc_to_invert, self.operator.components_[:pc_to_invert.shape[1], :]) + self.operator.mean_

        return inv


class CCAOps:

    def __init__(self):
        pass


class PosteriorOps:

    def __init__(self):
        self.posterior_mean = 0
        self.posterior_covariance = 0
        self.ops = DataOps()

    def posterior(self,
                  h_cca_training_gaussian,
                  d_cca_training,
                  d_pc_training,
                  d_rotations,
                  d_cca_prediction):
        """
        Estimating posterior uncertainties.
        @param h_cca_training_gaussian: Canonical Variate of the training target, Gaussian-distributed
        @param d_cca_training: Canonical Variate of the training data
        @param d_pc_training: Principal Components of the training data
        @param d_rotations: CCA rotations of the training data
        @param d_cca_prediction: Canonical Variate of the observation
        @return: h_mean_posterior, h_posterior_covariance
        """

        # TODO: add dimension check
        # Size of the set
        n_training = d_cca_training.shape[1]

        # Evaluate the covariance in h
        h_cov_operator = np.cov(h_cca_training_gaussian.T, rowvar=False)  # same

        # Evaluate the covariance in d (here we assume no data error, so C is identity times a given factor)
        x_dim = np.size(d_pc_training, axis=1)  # Number of PCA components for the curves
        noise = .01
        d_cov_operator = np.eye(x_dim) * noise  # I matrix.
        d_noise_covariance = d_rotations.T @ d_cov_operator @ d_rotations  # same

        # Linear modeling d to h
        g = np.linalg.lstsq(h_cca_training_gaussian.T, d_cca_training.T, rcond=None)[
            0].T  # Transpose to get same as Thomas
        g = np.where(np.abs(g) < 1e-12, 0, g)

        # Modeling error due to deviations from theory
        d_ls_predicted = g @ h_cca_training_gaussian  # same
        d_modeling_mean_error = np.mean(d_cca_training - d_ls_predicted, axis=1).reshape(-1, 1)  # same
        d_modeling_error = d_cca_training - d_ls_predicted - repmat(d_modeling_mean_error, 1,
                                                                    np.size(d_cca_training, axis=1))  # same
        # Information about the covariance of the posterior distribution.
        d_modeling_covariance = (d_modeling_error @ d_modeling_error.T) / n_training  # same

        # Computation of the posterior mean
        h_mean = np.row_stack(np.mean(h_cca_training_gaussian, axis=1))  # same
        h_mean = np.where(np.abs(h_mean) < 1e-12, 0, h_mean)  # My mean is 0, as expected.

        h_mean_posterior = h_mean \
                           + h_cov_operator @ g.T \
                           @ np.linalg.pinv(g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance) \
                           @ (d_cca_prediction.reshape(-1, 1) + d_modeling_mean_error - g @ h_mean)  # same

        # h posterior covariance
        h_posterior_covariance = h_cov_operator \
                                 - (h_cov_operator @ g.T) \
                                 @ np.linalg.pinv(g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance) \
                                 @ g @ h_cov_operator

        h_posterior_covariance = (h_posterior_covariance + h_posterior_covariance.T) / 2  # same

        self.posterior_mean = h_mean_posterior.T[0]
        self.posterior_covariance = h_posterior_covariance

        return h_mean_posterior.T[0], h_posterior_covariance

    # def random_sample(self,
    #                   h_cca_training,
    #                   d_cca_training,
    #                   d_pc_training,
    #                   d_rotations,
    #                   d_cca_prediction,
    #                   cca_obj, pca_obj, n_posts=1, add_comp=0):
    def random_sample(self, sample_n, pca_d, pca_h, cca_obj, n_posts=1, add_comp=0):

        # Cut desired number of PC components
        d_pc_training, d_pc_prediction = pca_d.pca_refresh(pca_d.ncomp)
        h_pc_training, h_pc_prediction = pca_h.pca_refresh(pca_h.ncomp)

        d_pc_obs = d_pc_prediction[sample_n]  # data for prediction sample
        h_pc_obs = h_pc_prediction[sample_n]  # target for prediction sample

        d_cca_training, h_cca_training = cca_obj.transform(d_pc_training, h_pc_training)
        d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T

        # Ensure Gaussian distribution in h_cca
        h_cca_training_gaussian = self.ops.gaussian_distribution(h_cca_training)

        # Get the rotation matrices
        d_rotations = cca_obj.x_rotations_

        # Project observed data into canonical space.
        d_cca_prediction, _ = cca_obj.transform(d_pc_obs.reshape(1, -1), h_pc_obs.reshape(1, -1))
        d_cca_prediction = d_cca_prediction.T

        # Estimate the posterior mean and covariance (Tarantola)
        h_mean_posterior, h_posterior_covariance = self.posterior(h_cca_training_gaussian,
                                                                  d_cca_training,
                                                                  d_pc_training,
                                                                  d_rotations,
                                                                  d_cca_prediction)
        shp = pca_h.raw_data.shape  # Original shape
        # n_posts = 500  # Number of estimates sampled from the distribution.
        # Draw n_posts random samples from the multivariate normal distribution :
        h_posts_gaussian = np.random.multivariate_normal(mean=self.posterior_mean,
                                                         cov=self.posterior_covariance,
                                                         size=n_posts).T
        # This h_posts gaussian need to be inverse-transformed to the original distribution.
        # We get the CCA scores.
        h_posts = self.ops.gaussian_inverse(h_posts_gaussian)
        # Calculate the values of hf, i.e. reverse the canonical correlation, it always works if dimf > dimh
        # The value of h_pca_reverse are the score of PCA in the forecast space.
        # To reverse data in the original space, perform the matrix multiplication between the data in the CCA space
        # with the y_loadings matrix. Because CCA scales the input, we must multiply the output by the y_std dev
        # and add the y_mean.
        h_pca_reverse = np.matmul(h_posts.T, cca_obj.y_loadings_.T) * cca_obj.y_std_ + cca_obj.y_mean_

        # Whether to add or not the rest of PC components
        if add_comp:
            rnpc = np.array([pca_h.pc_random(n_posts) for i in range(n_posts)])
            h_pca_reverse = np.array([np.concatenate((h_pca_reverse[i], rnpc[i])) for i in range(n_posts)])
        # Generate forecast in the initial dimension and reshape.
        forecast_ = pca_h.inverse_transform(h_pca_reverse).reshape((n_posts, shp[1], shp[2]))

        return forecast_


class Plot:

    def __init__(self):

        self.xlim = 1500
        self.ylim = 1000
        self.grf = 5
        self.nrow = self.ylim // self.grf
        self.ncol = self.xlim // self.grf
        self.x, self.y = np.meshgrid(
            np.linspace(0, self.xlim, int(self.xlim / self.grf)), np.linspace(0, self.ylim, int(self.ylim / self.grf)))
        self.wdir = jp(os.getcwd(), 'grid')
        self.cols = ['w', 'g', 'r', 'c', 'm', 'y']
        np.random.shuffle(self.cols)

    def curves(self, tc, n_wel, sdir=None, show=False):
        """
        Shows every breakthrough curve stacked on a plot.
        @param tc: Curves with shape (n_sim, n_wells, n_time_steps)
        @param n_wel: Number of observation points
        @param sdir: Directory in which to save figure
        @param show: Whether to show or not
        """
        for i in range(len(tc)):
            for t in range(n_wel):
                plt.plot(tc[i][t], color=self.cols[t], linewidth=.2, alpha=0.5)
        plt.grid(linewidth=.3, alpha=.4)
        plt.tick_params(labelsize=5)
        if sdir:
            plt.savefig(jp(sdir, 'curves.png'), dpi=300)
            plt.close()
        if show:
            plt.show()
            plt.close()

    def curves_i(self, tc, n_wel, sdir=None, show=False):
        """
        Shows every breakthrough individually for each observation point.
        Will produce n_well figures of n_sim curves each.
        @param tc: Curves with shape (n_sim, n_wells, n_time_steps)
        @param n_wel: Number of observation points
        @param sdir: Directory in which to save figure
        @param show: Whether to show or not
        """
        for t in range(n_wel):
            for i in range(len(tc)):
                plt.plot(tc[i][t], color=self.cols[t], linewidth=.2, alpha=0.5)
            plt.grid(linewidth=.3, alpha=.4)
            plt.tick_params(labelsize=5)
            if sdir:
                plt.savefig(jp(sdir, 'curves_{}.png'.format(t)), dpi=300)
                plt.close()
            if show:
                plt.show()
                plt.close()

    def whp(self,
            h,
            alpha=0.4,
            lw=.5,
            bkg_field_array=None,
            vmin=None,
            vmax=None,
            colors='white',
            fig_file=None,
            show=False):
        """
        Produces the WHPA plot, that is the zero-contour of the signed distance array.
        It assumes that well information can be loaded from pw.npy and iw.npy.
        I should change this.
        @param h:
        @param alpha:
        @param lw:
        @param colors:
        @param fig_file:
        @param show:
        @return:
        """
        # TODO: Add more options to customize the plot.
        # Plot background
        if bkg_field_array is not None:
            plt.imshow(bkg_field_array,
                       extent=(0, self.xlim, 0, self.ylim),
                       vmin=vmin,
                       vmax=vmax,
                       cmap='coolwarm')
            plt.colorbar()
        # Plot results
        for z in h:  # h is the n square WHPA matrix
            plt.contour(self.x, self.y, z, [0], colors=colors, linewidths=lw, alpha=alpha)
        plt.grid(color='c', linestyle='-', linewidth=.5, alpha=.2)
        # Plot wells
        pwl = np.load((jp(self.wdir, 'pw.npy')), allow_pickle=True)[:, :2]
        plt.plot(pwl[0][0], pwl[0][1], 'wo', label='pw')
        iwl = np.load((jp(self.wdir, 'iw.npy')), allow_pickle=True)[:, :2]
        for i in range(len(iwl)):
            plt.plot(iwl[i][0], iwl[i][1], 'o', markersize=4, markeredgecolor='k', markeredgewidth=.5,
                     label='iw{}'.format(i))
        plt.legend(fontsize=8)
        plt.xlim(750, 1200)
        plt.ylim(300, 700)
        plt.tick_params(labelsize=5)
        if fig_file:
            plt.savefig(fig_file, bbox_inches='tight', dpi=300)
            plt.close()
        if show:
            plt.show()
            plt.close()

    def whp_prediction(self, forecasts, h_true, h_pred, fig_file=None, show=False):
        self.whp(h=forecasts)
        # Plot true h
        plt.contour(self.x, self.y, h_true, [0], colors='red', linewidths=1, alpha=.9)
        # Plot true h predicted
        plt.contour(self.x, self.y, h_pred, [0], colors='cyan', linewidths=1, alpha=.9)
        if fig_file:
            plt.savefig(fig_file, bbox_inches='tight', dpi=300)
            plt.close()
        if show:
            plt.show()
            plt.close()

    def h_pca_inverse_plot(self, v, e, pca_o, vn):
        """
        Plot used to compare the reproduction of the original physical space after PCA transformation
        @param v: Original, untransformed signed distance array
        @param e: Sample number on which the test is performed
        @param pca_o: signed distance PCA operator
        @param vn: Number of components to inverse-transform.
        @return:
        """
        v_pc = pca_o.transform(v)
        v_pred = (np.dot(v_pc[e, :vn], pca_o.components_[:vn, :]) + pca_o.mean_)
        self.whp(v_pred.reshape(1, self.nrow, self.ncol), colors='cyan', alpha=.8, lw=1, show=False)
        self.whp(v[e].reshape(1, self.nrow, self.ncol), colors='red', alpha=1, lw=1, show=True)

    @staticmethod
    def d_pca_inverse_plot(v, e, pca_o, vn):
        """
        Plot used to compare the reproduction of the original physical space after PCA transformation
        @param v: Original, untransformed data array
        @param e: Sample number on which the test is performed
        @param pca_o: data PCA operator
        @param vn: Number of components to inverse-transform the data
        @return:
        """
        v_pc = pca_o.transform(v)
        v_pred = np.dot(v_pc[e, :vn], pca_o.components_[:vn, :]) + pca_o.mean_
        plt.plot(v[e], 'r', alpha=.8)
        plt.plot(v_pred, 'c', alpha=.8)
        plt.show()

    @staticmethod
    def explained_variance(pca, n_comp=0, xfs=2, fig_file=None, show=False):
        """
        PCA explained variance plot
        @param pca: PCA operator
        @param n_comp: Number of components to display
        @param xfs: X-axis fontsize
        @param fig_file:
        @param show:
        @return:
        """
        plt.grid(alpha=0.2)
        if not n_comp:
            n_comp = pca.n_components_
        plt.xticks(np.arange(n_comp), fontsize=xfs)
        plt.plot(np.arange(n_comp), np.cumsum(pca.explained_variance_ratio_[:n_comp]),
                 '-o', linewidth=.5, markersize=1.5, alpha=.8)
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        if fig_file:
            plt.savefig(fig_file, dpi=300)
            plt.close()
        if show:
            plt.show()
            plt.close()

    @staticmethod
    def pca_scores(training, prediction, n_comp, fig_file=None, show=False):
        """
        PCA scores plot, displays scores of observations above those of training.
        @param training: Training scores
        @param prediction: Test scores
        @param n_comp: How many componnents to show
        @param fig_file:
        @param show:
        @return:
        """
        # Scores plot
        plt.grid(alpha=0.2)
        ut = n_comp
        plt.xticks(np.arange(ut), fontsize=8)
        plt.plot(training.T[:ut], 'wo', markersize=1, alpha=0.2)  # Plot all training scores
        for sample_n in range(len(prediction)):
            pc_obs = prediction[sample_n]
            plt.plot(pc_obs.T[:ut],  # Plot observations scores
                     'o', markersize=2.5, markeredgecolor='k', markeredgewidth=.4, alpha=.8,
                     label=str(sample_n))
        plt.tick_params(labelsize=6)
        plt.legend(fontsize=3)

        if fig_file:
            plt.savefig(fig_file, dpi=300)
            plt.close()
        if show:
            plt.show()
            plt.close()

    @staticmethod
    def cca(cca, d, h, d_pc_prediction, h_pc_prediction, sdir=None, show=False):
        """
        CCA plots.
        Receives d, h PC components to be predicted, transforms them in CCA space and adds it to the plots.
        @param cca: CCA operator
        @param d: d CCA scores
        @param h: h CCA scores
        @param d_pc_prediction: d test PC scores
        @param h_pc_prediction: h test PC scores
        @param sdir:
        @param show:
        @return:
        """

        cca_coefficient = np.corrcoef(d, h).diagonal(offset=cca.n_components)  # Gets correlation coefficient

        # CCA plots for each observation:
        for i in range(cca.n_components):
            comp_n = i
            plt.plot(d[comp_n], h[comp_n], 'ro', markersize=3, markerfacecolor='r', alpha=.25)
            for sample_n in range(len(d_pc_prediction)):  # For each 'observation'
                d_obs = d_pc_prediction[sample_n]
                h_obs = h_pc_prediction[sample_n]
                d_cca_prediction, h_cca_prediction = cca.transform(d_obs.reshape(1, -1),
                                                                   h_obs.reshape(1, -1))
                d_cca_prediction, h_cca_prediction = d_cca_prediction.T, h_cca_prediction.T

                plt.plot(d_cca_prediction[comp_n], h_cca_prediction[comp_n],
                         'o', markersize=4.5, alpha=.7,
                         label='{}'.format(sample_n))

            plt.grid('w', linewidth=.3, alpha=.4)
            plt.tick_params(labelsize=8)
            plt.title(round(cca_coefficient[i], 4))
            plt.legend(fontsize=5)
            if sdir:
                plt.savefig(jp(sdir, 'cca{}.png'.format(i)), bbox_inches='tight', dpi=300)
                plt.close()
            if show:
                plt.show()
                plt.close()
