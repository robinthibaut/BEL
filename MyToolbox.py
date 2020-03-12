import os
from os.path import join as jp
import shutil

import flopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d


class FileOps:

    def __init__(self):
        pass

    @staticmethod
    def load_data(res_dir, n=0):
        bkt_files = []
        sd_files = []
        hk_files = []
        # r=root, d=directories, f = files
        roots = []
        for r, d, f in os.walk(res_dir, topdown=False):
            if 'bkt.npy' in f and 'sd.npy' in f and 'hk0.npy' in f:
                bkt_files.append(os.path.join(r, 'bkt.npy'))
                sd_files.append(os.path.join(r, 'sd.npy'))
                hk_files.append(os.path.join(r, 'hk0.npy'))
                roots.append(r)
            else:
                try:
                    if r != res_dir:
                        shutil.rmtree(r)
                except TypeError:
                    pass
        if n:
            folders = np.random.choice(np.arange(len(roots)), n)
            bkt_files = np.array(bkt_files)[folders]
            sd_files = np.array(sd_files)[folders]
            hk_files = np.array(hk_files)[folders]
            roots = np.array(roots)[folders]

        # Load and filter results
        tpt = list(map(np.load, bkt_files))
        # FIXME: the way transport arrays are saved
        rm = []
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

        tpt = list(map(np.load, bkt_files))  # Re-load transport curves
        sd = np.array(list(map(np.load, sd_files)))  # Load signed distance
        hk = np.array(list(map(np.load, hk_files)))  # Load hydraulic K

        return tpt, sd, hk

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
                x0)  # Cumulation of width should equal xlim, but it will not be the case, have to adapt width
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

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)

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
        pass

    @staticmethod
    def d_process(tc0, n_time_steps=500, t_max=1.01080e+02):
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


class Plot:

    def __init__(self):

        self.xlim = 1500
        self.ylim = 1000
        self.grf = 5
        self.x, self.y = np.meshgrid(
            np.linspace(0, self.xlim, int(self.xlim / self.grf)), np.linspace(0, self.ylim, int(self.ylim / self.grf)))

        self.cols = ['w', 'g', 'r', 'c', 'm', 'y']
        np.random.shuffle(self.cols)

    def curves(self, tc, n_wel, sdir=None, show=False):
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

    def whp(self, h, wdir, fig_file=None, show=False):
        # Plot results
        for z in h:
            plt.contour(self.x, self.y, z, [0], colors='white', linewidths=.5, alpha=0.4)
        plt.grid(color='c', linestyle='-', linewidth=.5, alpha=0.4)
        # Plot wells
        pwl = np.load((jp(wdir, 'pw.npy')), allow_pickle=True)[:, :2]
        plt.plot(pwl[0][0], pwl[0][1], 'wo', label='pw')
        iwl = np.load((jp(wdir, 'iw.npy')), allow_pickle=True)[:, :2]
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

    def whp_prediction(self, forecasts, h_true, h_pred, wdir, fig_file=None, show=False):
        self.whp(h=forecasts, wdir=wdir)
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

    @staticmethod
    def evr_plot(pca):
        plt.plot(np.arange(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        plt.show()
        plt.close()

    @staticmethod
    def cca_plot(cca_coef, cc, sdc, obs_n, j, x_c_pred, y_c_true):
        plt.plot(cc[:, j], sdc[:, j], 'ro')
        plt.axvline(x_c_pred[obs_n, j])
        plt.plot(x_c_pred[obs_n, j], y_c_true[obs_n, j], 'go')
        plt.xlabel('$d_{}$'.format(j))
        plt.ylabel('$h_{}$'.format(j))
        plt.title('corr = {}'.format(round(cca_coef[j], 3)))

    @staticmethod
    def cca_plots(coeffs, d_cca_t, h_cca_t, d_cca_p, h_cca_p):
        n_comp = len(coeffs)
        fig, axs = plt.subplots(nrows=n_comp % (3 % n_comp) + 1,
                                ncols=3 % n_comp + 1,
                                figsize=(9, 6),
                                subplot_kw={'xticks': [], 'yticks': []})
        for i in range(n_comp):
            axs.flat[i].plot(d_cca_t[i], h_cca_t[i], 'ro')
            axs.flat[i].plot(d_cca_p[i], h_cca_p[i], 'ko')
            axs.flat[i].set_title(str(np.round(coeffs[i], 3)))
        plt.tight_layout()
        plt.show()
        plt.close()

    @staticmethod
    def explained_variance(pca, n_comp=0, xfs=2, fig_file=None, show=False):
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
        # Scores plot
        plt.grid(alpha=0.2)
        ut = n_comp
        plt.xticks(np.arange(ut), fontsize=8)
        plt.plot(training[:ut], 'wo', markersize=1, alpha=0.6)
        for sample_n in range(len(prediction)):
            pc_obs = prediction[sample_n]
            plt.plot(pc_obs[:ut],
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

        cca_coefficient = np.corrcoef(d, h).diagonal(offset=cca.n_components)
        # CCA plots for each observation:

        for i in range(cca.n_components):
            comp_n = i
            plt.plot(d[comp_n], h[comp_n], 'ro', markersize=3, markerfacecolor='r', alpha=.25)
            for sample_n in range(len(d_pc_prediction)):
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

# class MyWorkshop:
#
#     def __init__(self):
#         self.xlim = 1500
#         self.ylim = 1000
#         self.grf = 1
#         self.ntstp = 1000
#         self.res_dir = ''
#
#     def load_data(self):
#         bkt_files = []
#         sd_files = []
#         hk_files = []
#         # r=root, d=directories, f = files
#         roots = []
#         for r, d, f in os.walk(self.res_dir, topdown=False):
#             if 'bkt.npy' in f and 'sd.npy' in f and 'hk0.npy' in f:
#                 bkt_files.append(os.path.join(r, 'bkt.npy'))
#                 sd_files.append(os.path.join(r, 'sd.npy'))
#                 hk_files.append(os.path.join(r, 'hk0.npy'))
#                 roots.append(r)
#             else:
#                 try:
#                     if r != self.res_dir:
#                         shutil.rmtree(r)
#                 except TypeError:
#                     pass
#
#         # Load and filter results
#         tpt = list(map(np.load, bkt_files))
#         # FIXME: the way transport arrays are saved
#         # n_wells = 50
#         # tpt = [np.split(t, n_wells) for t in tpt]
#
#         rm = []
#         for i in range(len(tpt)):
#             for j in range(len(tpt[i])):
#                 if max(tpt[i][j][:, 1]) > 1:  # Check results files whose max computed head is > 1 and removes them
#                     rm.append(i)
#                     break
#         for index in sorted(rm, reverse=True):
#             del bkt_files[index]
#             del sd_files[index]
#             del hk_files[index]
#             del roots[index]
#
#         tpt = list(map(np.load, bkt_files))  # Re-load transport curves
#         # tpt = [np.split(t, n_wells) for t in tpt]
#
#         # ny = []  # Smoothed curves
#         # tstp = []  # Time steps, which are different for each simulation
#         #
#         # for c in tpt:  # First, smoothing of the curves with Savgol filter
#         #     for ob in c:
#         #         tstp.append(c[0][:, 0])
#         #         v = ob[:, 1]
#         #         ny.append(np.array(savgol_filter(v, 51, 3)))  # smoothed res
#         #
#         # flat_ny = [item for sublist in ny for item in sublist]
#         # max_v = max(flat_ny)  # Max and min value to normalize
#         # min_v = min(flat_ny)
#         #
#         # f1d = []  # List of interpolating functions for each curve
#         # for i in range(len(ny)):
#         #     o_s = (ny[i] - min_v) / (max_v - min_v)
#         #     f1d.append(interp1d(tstp[i], o_s, fill_value='extrapolate'))
#         #
#         # # ntstp = 1000  # Arbitrary number of time steps to create the final transport array
#         # ls = np.linspace(0, 200, num=self.ntstp)  # From 0 to 200 days with 1000 steps
#         #
#         # tc = []
#         # [tc.append(f1d[i](ls)) for i in range(len(f1d))]
#         # tc = np.array(tc)
#         # tc = tc.reshape((len(tpt), len(tpt[0]), self.ntstp))
#
#         sd = np.array(list(map(np.load, sd_files)))  # Load signed distance
#         hk = np.array(list(map(np.load, hk_files)))  # Load hydraulic K
#
#         # return tpt, tc, max_v, min_v, sd, hk
#         return tpt, sd, hk
#
#     def protection_zone_plot(self, pz):
#         """Displays every protection zones (0 contour line) contained in the pz array"""
#         X_pz, Y_pz = np.meshgrid(np.linspace(0, self.xlim, int(self.xlim / self.grf)),
#                                  np.linspace(0, self.ylim, int(self.ylim / self.grf)))
#         for s in range(len(pz)):
#             plt.contour(X_pz, Y_pz, pz[s], [0], colors='blue', alpha=0.5)
#         # plt.imshow(Z, origin='lower',
#         #            cmap='coolwarm', alpha=0.5)
#         # plt.colorbar()
#         # plt.xlim(600, 1200)
#         # plt.ylim(200, 800)
#         # plt.savefig(jp(cwd, 'signed_distance_c.png'), dpi=300, bbox_inches='tight')
#         plt.show()
#
#     def hydraulic_conductivity_plot(self, hka):
#         for k in range(len(hka)):
#             # t = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
#             hkp = plt.imshow(hka[k][0], extent=(0, self.xlim, 0, self.ylim), cmap='coolwarm',
#                              norm=LogNorm(vmin=hka[k][0].min(), vmax=hka[k][0].max()))
#             plt.colorbar()
#             # contours = plt.contour(X, Y, sd[i], [0], colors='black', alpha=0.7)
#             # plt.savefig(jp(res_dir, 'signed_distance_{}.png'.format(i)), dpi=150, bbox_inches='tight')
#             plt.show()
#             plt.close()
