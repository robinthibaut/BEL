#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline

from experiment.base.inventory import MySetup
from experiment.toolbox import filesio

plt.style.use('dark_background')


def explained_variance(pca, n_comp=0, xfs=2, thr=1., fig_file=None, show=False):
    """
    PCA explained variance plot
    :param pca: PCA operator
    :param n_comp: Number of components to display
    :param xfs: X-axis fontsize
    :param fig_file:
    :param show:
    :return:
    """
    plt.grid(alpha=0.1)
    if not n_comp:
        n_comp = pca.n_components_

    # plt.xticks(np.arange(n_comp), fontsize=xfs)
    ny = len(np.where(np.cumsum(pca.explained_variance_ratio_) < thr)[0])
    cum = np.cumsum(pca.explained_variance_ratio_[:n_comp]) * 100
    yticks = np.append(cum[:ny], cum[-1])
    plt.yticks(yticks)
    plt.bar(np.arange(n_comp), np.cumsum(pca.explained_variance_ratio_[:n_comp]) * 100, color='m', alpha=.1)
    plt.plot(np.arange(n_comp), np.cumsum(pca.explained_variance_ratio_[:n_comp]) * 100,
             '-o', linewidth=.5, markersize=1.5, alpha=.8)
    plt.xlabel('Components number')
    plt.ylabel('Cumulative explained variance (%)')
    if fig_file:
        filesio.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, dpi=300, transparent=True)
        plt.close()
    if show:
        plt.show()
        plt.close()


def pca_scores(training, prediction, n_comp, fig_file=None, labels=True, show=False):
    """
    PCA scores plot, displays scores of observations above those of training.
    :param training: Training scores
    :param prediction: Test scores
    :param n_comp: How many components to show
    :param fig_file:
    :param show:
    :return:
    """
    # Scores plot
    # Grid
    plt.grid(alpha=0.2)
    # Number of components to keep
    ut = n_comp
    # Ticks
    plt.xticks(np.arange(0, ut, 5), fontsize=8)
    # Plot all training scores
    plt.plot(training.T[:ut], 'ow', markersize=3, alpha=0.05)
    # plt.plot(training.T[:ut], '+w', markersize=.5, alpha=0.2)
    # Choose seaborn cmap
    # cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=.69, reverse=True)
    # cmap = sns.cubehelix_palette(start=6, rot=0, dark=0, light=.69, reverse=True, as_cmap=True)
    # cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=.15, reverse=True)
    # KDE plot
    # sns.kdeplot(np.arange(1, ut + 1), training.T[:ut][:, 1], cmap=cmap, n_levels=60, shade=True, vertical=True)
    plt.xlim(0.5, ut)  # Define x limits [start, end]
    # For each sample used for prediction:
    for sample_n in range(len(prediction)):
        pc_obs = prediction[sample_n]

        # Crete beautiful spline to follow prediction scores
        xnew = np.linspace(1, ut, 200)
        spl = make_interp_spline(np.arange(1, ut + 1), pc_obs.T[:ut], k=3)  # type: BSpline
        power_smooth = spl(xnew)
        plt.plot(xnew - 1, power_smooth, 'red', linewidth=1.2, alpha=.9)
        # plt.plot(xnew - 1, power_smooth, 'y', linewidth=.3, alpha=.5)

        plt.plot(pc_obs.T[:ut],  # Plot observations scores
                 'ro', markersize=3, markeredgecolor='k', markeredgewidth=.4, alpha=.8,
                 label=str(sample_n))
        # plt.plot(pc_obs.T[:ut],  # Plot observations scores
        #          'o', markersize=2.5, markeredgecolor='k', markeredgewidth=.4, alpha=.8,
        #          label=str(sample_n))

    if labels:
        plt.title('PCA scores')
        plt.xlabel('Component id')
        plt.ylabel('Component scores')
    plt.tick_params(labelsize=6)

    if fig_file:
        filesio.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, dpi=300, transparent=True)
        plt.close()
    if show:
        plt.show()
        plt.close()


def cca_plot(cca_operator, d, h, d_pc_prediction, h_pc_prediction, sdir=None, show=False):
    """
    CCA plots.
    Receives d, h PC components to be predicted, transforms them in CCA space and adds it to the plots.
    :param cca_operator: CCA operator
    :param d: d CCA scores
    :param h: h CCA scores
    :param d_pc_prediction: d test PC scores
    :param h_pc_prediction: h test PC scores
    :param sdir:
    :param show:
    :return:
    """

    cca_coefficient = np.corrcoef(d, h).diagonal(offset=cca_operator.n_components)  # Gets correlation coefficient

    # CCA plots for each observation:
    for i in range(cca_operator.n_components):
        comp_n = i
        # plt.plot(d[comp_n], h[comp_n], 'w+', markersize=3, markerfacecolor='w', alpha=.25)
        for sample_n in range(len(d_pc_prediction)):  # For each 'observation'
            # Extract from sample
            d_obs = d_pc_prediction[sample_n]
            h_obs = h_pc_prediction[sample_n]
            # Transform to CCA space
            d_cca_prediction, h_cca_prediction = cca_operator.transform(d_obs.reshape(1, -1),
                                                                        h_obs.reshape(1, -1))
            d_cca_prediction, h_cca_prediction = d_cca_prediction.T, h_cca_prediction.T

            cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=.95, reverse=True)
            g = sns.jointplot(d[comp_n], h[comp_n],
                              cmap=cmap, n_levels=80, shade=True,
                              kind='kde')
            g.plot_joint(plt.scatter, c='w', marker='+', s=2, alpha=.7)
            # add 'arrows' at observation location
            g.ax_marg_x.arrow(d_cca_prediction[comp_n], 0, 0, .1)
            g.ax_marg_y.arrow(0, h_cca_prediction[comp_n], .1, 0)
            plt.plot(d_cca_prediction[comp_n], h_cca_prediction[comp_n],
                     'wo', markersize=4.5, markeredgecolor='k', alpha=.7,
                     label='{}'.format(sample_n))
        # plt.grid('w', linewidth=.3, alpha=.4)
        # plt.tick_params(labelsize=8)
        plt.xlabel('d')
        plt.ylabel('h')
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f'{i} - {round(cca_coefficient[i], 4)}')
        if sdir:
            filesio.dirmaker(sdir)
            plt.savefig(jp(sdir, 'cca{}.png'.format(i)), bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
        if show:
            plt.show()
            plt.close()

    # # CCA plots for each observation:
    # for i in range(cca_operator.n_components):
    #     comp_n = i
    #     plt.plot(d[comp_n], h[comp_n], 'ro', markersize=3, markerfacecolor='r', alpha=.25)
    #     for sample_n in range(len(d_pc_prediction)):  # For each 'observation'
    #         d_obs = d_pc_prediction[sample_n]
    #         h_obs = h_pc_prediction[sample_n]
    #         d_cca_prediction, h_cca_prediction = cca_operator.transform(d_obs.reshape(1, -1),
    #                                                                     h_obs.reshape(1, -1))
    #         d_cca_prediction, h_cca_prediction = d_cca_prediction.T, h_cca_prediction.T
    #
    #         plt.plot(d_cca_prediction[comp_n], h_cca_prediction[comp_n],
    #                  'o', markersize=4.5, alpha=.7,
    #                  label='{}'.format(sample_n))
    #
    #     plt.grid('w', linewidth=.3, alpha=.4)
    #     plt.tick_params(labelsize=8)
    #     plt.title(round(cca_coefficient[i], 4))
    #     plt.legend(fontsize=5)
    #     if sdir:
    #         plt.savefig(jp(sdir, 'cca{}.png'.format(i)), bbox_inches='tight', dpi=300)
    #         plt.close()
    #     if show:
    #         plt.show()
    #         plt.close()


class Plot:

    def __init__(self, x_lim=None, y_lim=None, grf=None, wel_comb=None):

        md = MySetup.Directories()
        focus = MySetup.Focus()
        self.wels = MySetup.Wels()

        if wel_comb is not None:
            self.wels.combination = wel_comb

        if y_lim is None:
            self.ylim = focus.y_range
        else:
            self.ylim = y_lim
        if x_lim is None:
            self.xlim = focus.x_range
        else:
            self.xlim = x_lim
        if grf is None:
            self.grf = focus.cell_dim
        else:
            self.grf = grf

        self.nrow = int(np.diff(self.ylim) / self.grf)  # Number of rows
        self.ncol = int(np.diff(self.xlim) / self.grf)  # Number of columns
        self.x, self.y = np.meshgrid(
            np.linspace(self.xlim[0], self.xlim[1], self.ncol), np.linspace(self.ylim[0], self.ylim[1], self.nrow))
        self.wdir = md.grid_dir
        # self.cols = self.wels.wels_data
        wells_id = list(self.wels.wels_data.keys())
        self.cols = [self.wels.wels_data[w]['color'] for w in wells_id if 'pumping' not in w]

    def contours_vertices(self, arrays, c=0):
        """
        Extracts contour vertices from a list of matrices
        :param arrays: list of matrices
        :param c: Contour value
        :return: vertices array
        """
        if len(arrays.shape) < 3:
            arrays = [arrays]
        # First create figures for each forecast.
        c0s = [plt.contour(self.x, self.y, f, [c]) for f in arrays]
        plt.close()  # Close plots
        # .allseg[0][0] extracts the vertices of each O contour = WHPA's vertices
        v = np.array([c0.allsegs[0][0] for c0 in c0s], dtype=object)
        return v

    def curves(self, tc, highlight=None, sdir=None, show=False):
        """
        Shows every breakthrough curve stacked on a plot.
        :param tc: Curves with shape (n_sim, n_wels, n_time_steps)
        :param highlight: list: List of indices of curves to highlight in the plot
        :param sdir: Directory in which to save figure
        :param show: Whether to show or not
        """
        if highlight is None:
            highlight = []
        title = 'curves'
        n_sim, n_wels, nts = tc.shape
        for i in range(n_sim):
            for t in range(n_wels):
                if i in highlight:
                    plt.plot(tc[i][t], color=self.cols[t], linewidth=2, alpha=1)
                else:
                    plt.plot(tc[i][t], color=self.cols[t], linewidth=.2, alpha=0.5)

        plt.grid(linewidth=.3, alpha=.4)
        plt.tick_params(labelsize=5)
        if sdir:
            filesio.dirmaker(sdir)
            plt.savefig(jp(sdir, f'{title}.png'), dpi=300, transparent=True)
            plt.close()
        if show:
            plt.show()
            plt.close()

    def curves_i(self, tc, highlight=None, sdir=None, show=False):
        """
        Shows every breakthrough individually for each observation point.
        Will produce n_well figures of n_sim curves each.
        :param tc: Curves with shape (n_sim, n_wells, n_time_steps)
        :param highlight: list: List of indices of curves to highlight in the plot
        :param sdir: Directory in which to save figure
        :param show: Whether to show or not
        """
        if highlight is None:
            highlight = []
        title = 'curves'
        n_sim, n_wels, nts = tc.shape
        for t in range(n_wels):
            for i in range(n_sim):
                if i in highlight:
                    plt.plot(tc[i][t], color=self.cols[t], linewidth=2, alpha=1)
                else:
                    plt.plot(tc[i][t], color=self.cols[t], linewidth=.2, alpha=0.5)
            plt.grid(linewidth=.3, alpha=.4)
            plt.tick_params(labelsize=5)
            plt.title(f'wel #{t + 1}')
            if sdir:
                filesio.dirmaker(sdir)
                plt.savefig(jp(sdir, f'{title}_{t + 1}.png'), dpi=300, transparent=True)
                plt.close()
            if show:
                plt.show()
                plt.close()

    def whp(self,
            h=None,
            alpha=0.4,
            lw=.5,
            bkg_field_array=None,
            vmin=None,
            vmax=None,
            x_lim=None,
            y_lim=None,
            cmap='coolwarm',
            colors='white',
            show_wells=False,
            title=None,
            fig_file=None,
            show=False):
        """
        Produces the WHPA plot, i.e. the zero-contour of the signed distance array.

        :param title: str: plot title
        :param show_wells: bool: whether to plot well coordinates or not
        :param cmap: str: colormap name for the background array
        :param vmax: float: max value to plot for the background array
        :param vmin: float: max value to plot for the background array
        :param bkg_field_array: np.array: 2D array whose values will be plotted on the grid
        :param h: np.array: Array containing grids of values whose 0 contour will be computed and plotted
        :param alpha: float: opacity of the 0 contour lines
        :param lw: float: Line width
        :param colors: str: Line color
        :param fig_file: str:
        :param show: bool:
        :param x_lim: [x_min, x_max]
        :param y_lim: [y_min, y_max]
        """

        # Plot background
        if bkg_field_array is not None:
            plt.imshow(bkg_field_array,
                       extent=(self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]),
                       vmin=vmin,
                       vmax=vmax,
                       cmap=cmap)
            plt.colorbar()

        # Plot results
        if h is None:
            h = []

        for z in h:  # h is the n square WHPA matrix
            plt.contour(self.x, self.y, z, [0], colors=colors, linewidths=lw, alpha=alpha)
        plt.grid(color='c', linestyle='-', linewidth=.5, alpha=.2)

        # Plot wells
        if show_wells:
            comb = [0] + list(self.wels.combination)
            keys = [list(self.wels.wels_data.keys())[i] for i in comb]
            wbd = {k: self.wels.wels_data[k] for k in keys if k in self.wels.wels_data}
            # Get pumping well coordinates
            pwl = wbd['pumping0']['coordinates']
            plt.plot(pwl[0], pwl[1], 'wo', label='pw')
            for n, i in enumerate(wbd):
                plt.plot(wbd[i]['coordinates'][0], wbd[i]['coordinates'][1],
                         f'{wbd[i]["color"]}o', markersize=4, markeredgecolor='k', markeredgewidth=.5,
                         label=f'{n}')
            plt.legend(fontsize=8)

        # Plot limits
        if x_lim is None:
            plt.xlim(self.xlim[0], self.xlim[1])
        else:
            plt.xlim(x_lim[0], x_lim[1])
        if y_lim is None:
            plt.ylim(self.ylim[0], self.ylim[1])
        else:
            plt.ylim(y_lim[0], y_lim[1])

        if title:
            plt.title(title)

        # Tick size
        plt.tick_params(labelsize=5)

        if fig_file:
            filesio.dirmaker(os.path.dirname(fig_file))
            plt.savefig(fig_file, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
        if show:
            plt.show()
            plt.close()

    def whp_prediction(self,
                       forecasts,
                       h_true,
                       h_pred=None,
                       bkg_field_array=None,
                       fig_file=None,
                       show_wells=False,
                       title=None,
                       show=False):

        self.whp(h=forecasts,
                 show_wells=show_wells,
                 bkg_field_array=bkg_field_array,
                 title=title)
        # Plot true h
        plt.contour(self.x, self.y, h_true, [0], colors='red', linewidths=1, alpha=.9)

        # Plot 'true' h predicted
        if h_pred is not None:
            plt.contour(self.x, self.y, h_pred, [0], colors='cyan', linewidths=1, alpha=.9)
        if fig_file:
            filesio.dirmaker(os.path.dirname(fig_file))
            plt.savefig(fig_file, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
        if show:
            plt.show()
            plt.close()

    @staticmethod
    def d_pca_inverse_plot(pca_o, vn, training=True, fig_dir=None, show=False):
        """

        Plot used to compare the reproduction of the original physical space after PCA transformation.

        :param pca_o: data PCA operator
        :param vn: Number of components to keep while inverse-transforming the data
        :param training: bool:
        :param fig_dir: str:
        :param show: bool:
        :return:
        """

        if training:
            v_pc = pca_o.training_pc
            roots = pca_o.roots
        else:
            v_pc = pca_o.predict_pc
            roots = pca_o.test_root

        for i, r in enumerate(roots):
            v_pred = np.dot(v_pc[i, :vn], pca_o.operator.components_[:vn, :]) + pca_o.operator.mean_
            if training:
                plt.plot(pca_o.training_physical[i], 'r', alpha=.8)
            else:
                plt.plot(pca_o.predict_physical[i], 'r', alpha=.8)
            plt.plot(v_pred, 'c', alpha=.8)
            if fig_dir is not None:
                filesio.dirmaker(fig_dir)
                plt.savefig(jp(fig_dir, f'{r}_d.png'), dpi=100, transparent=True)
                plt.close()
            if show:
                plt.show()
                plt.close()

    def h_pca_inverse_plot(self, pca_o, vn, training=True, fig_dir=None, show=False):
        """

        Plot used to compare the reproduction of the original physical space after PCA transformation

        :param pca_o: signed distance PCA operator
        :param vn: Number of components to keep while inverse-transforming the data
        :param training: bool:
        :param fig_dir: str:
        :param show: bool:
        :return:
        """

        shape = pca_o.shape

        if training:
            v_pc = pca_o.training_pc
            roots = pca_o.roots
        else:
            v_pc = pca_o.predict_pc
            roots = pca_o.test_root

        for i, r in enumerate(roots):
            v_pred = (np.dot(v_pc[i, :vn], pca_o.operator.components_[:vn, :]) + pca_o.operator.mean_)
            self.whp(h=v_pred.reshape(1, shape[1], shape[2]), colors='cyan', alpha=.8, lw=1)
            if training:
                self.whp(h=pca_o.training_physical[i].reshape(1, shape[1], shape[2]), colors='red', alpha=1, lw=1)
            else:
                self.whp(h=pca_o.predict_physical[i].reshape(1, shape[1], shape[2]), colors='red', alpha=1, lw=1)
            if fig_dir is not None:
                filesio.dirmaker(fig_dir)
                plt.savefig(jp(fig_dir, f'{r}_h.png'), dpi=300, transparent=True)
                plt.close()
            if show:
                plt.show()
                plt.close()

    def pca_inverse_compare(self, pco_d=None, pco_h=None, nd=None, nh=None, show=False):
        """
        Plots original data and recovered data by PCA inverse transformation given a number of components
        :param pco_h: PCA object for h
        :param pco_d: PCA object for d
        :param nd: number of components in d
        :param nh: number of components in h
        :return:
        """

        for n_compare, r in enumerate(pco_h.roots):
            fig_dir = jp(MySetup.Directories.forecasts_dir, 'base', 'control')
            filesio.dirmaker(fig_dir)

            fig_file = jp(fig_dir, r)

            if pco_d is not None:
                self.d_pca_inverse_plot(pco_d.training_physical,
                                        n_compare,
                                        pco_d.operator,
                                        nd,
                                        fig_file=''.join((fig_file, '_d')), show=show)

            if pco_h is not None:
                self.h_pca_inverse_plot(pco_h,
                                        n_compare,
                                        nh,
                                        fig_file=''.join((fig_file, '_h')), show=show)

        # Displays the explained variance percentage given the number of components
        # print(pco_d.perc_pca_components(nd))
        # print(pco_h.perc_pca_components(nh))

    def check_root(self, root):
        """
        Plots raw data of folder 'root'
        :param root:
        :return:
        """
        bkt, whpa, _ = filesio.load_res(roots=root)
        whpa = whpa.squeeze()
        # self.curves_i(bkt, show=True)  # This function will not work

        plt.plot(whpa[:, 0], whpa[:, 1], 'wo')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.show()

    def plot_results(self, root, folder):
        """
        Plots forecasts results in the 'uq' folder
        :param root: str: Forward ID
        :param folder: str: Well combination. '123456', '1'...
        :return:
        """
        # Directory
        md = jp(MySetup.Directories.forecasts_dir, root, folder)
        # CCA pickle
        cca_operator = joblib.load(jp(md, 'obj', 'cca.pkl'))
        # h PCA pickle
        hbase = jp(MySetup.Directories.forecasts_dir, 'base')
        pcaf = jp(hbase, 'h_pca.pkl')
        h_pco = joblib.load(pcaf)

        # Curves - d
        # Plot curves
        sdir = jp(md, 'data')
        d_pco = joblib.load(jp(md, 'obj', 'd_pca.pkl'))
        tc = d_pco.training_physical.reshape(d_pco.training_shape)
        tcp = d_pco.predict_physical.reshape(d_pco.obs_shape)
        tc = np.concatenate((tc, tcp), axis=0)
        self.curves(tc=tc, sdir=sdir, highlight=[len(tc) - 1])
        self.curves_i(tc=tc, sdir=sdir, highlight=[len(tc) - 1])

        # WHP - h
        fig_dir = jp(hbase, 'roots_whpa')
        ff = jp(fig_dir, f'{root}.png')  # figure name
        h = np.load(jp(md, 'obj', 'h_true_obs.npy')).reshape(h_pco.obs_shape)
        h_training = h_pco.training_physical.reshape(h_pco.training_shape)
        # Plots target training + prediction
        self.whp(h_training, alpha=.2, show=False)
        self.whp(h, colors='r', lw=1, alpha=1, fig_file=ff)

        # WHPs
        ff = jp(md,
                'uq',
                f'cca_{cca_operator.n_components}.png')
        h_training = h_pco.training_physical.reshape(h_pco.training_shape)
        # forecast_posterior = np.load(jp(md, 'obj', 'forecast_posterior.npy'))
        post_obj = joblib.load(jp(md, 'obj', 'post.pkl'))
        forecast_posterior = post_obj.random_sample(pca_d=d_pco,
                                                    pca_h=h_pco,
                                                    cca_obj=cca_operator,
                                                    n_posts=MySetup.Forecast.n_posts,
                                                    add_comp=False)
        h_true_obs = np.load(jp(md, 'obj', 'h_true_obs.npy'))
        # h_pred = np.load(jp(md, 'obj', 'h_pred.npy'))

        self.whp(h_training, lw=.1, alpha=.1, colors='b', show=False)
        self.whp_prediction(forecasts=forecast_posterior,
                            h_true=h_true_obs,
                            # h_pred=h_pred,
                            show_wells=True,
                            fig_file=ff)

    @staticmethod
    def pca_vision(root, d=True, h=False, scores=True, exvar=True, folders=None):
        """
        Loads PCA pickles and plot scores for all folders
        :param root: str:
        :param d: bool:
        :param h: bool:
        :param scores: bool:
        :param exvar: bool:
        :param folders: list:
        :return:
        """

        if isinstance(root, (list, tuple)):
            if len(root) > 1:
                print('Input error')
                return
            else:
                root = root[0]

        subdir = os.path.join(MySetup.Directories.forecasts_dir, root)
        if folders is None:
            listme = os.listdir(subdir)
            folders = list(filter(lambda du: os.path.isdir(os.path.join(subdir, du)), listme))
        else:
            if not isinstance(folders, (list, tuple)):
                folders = [folders]

        if d:
            for f in folders:
                dfig = os.path.join(subdir, f, 'pca')
                # For d only
                pcaf = os.path.join(subdir, f, 'obj', 'd_pca.pkl')
                d_pco = joblib.load(pcaf)
                fig_file = os.path.join(dfig, 'd_scores.png')
                if scores:
                    pca_scores(training=d_pco.training_pc,
                               prediction=d_pco.predict_pc,
                               n_comp=d_pco.ncomp,
                               labels=False,
                               fig_file=fig_file)
                # Explained variance plots
                if exvar:
                    fig_file = os.path.join(dfig, 'd_exvar.png')
                    explained_variance(d_pco.operator, n_comp=d_pco.ncomp, thr=.9, fig_file=fig_file)
        if h:
            hbase = os.path.join(MySetup.Directories.forecasts_dir, 'base')
            # Load h pickle
            pcaf = os.path.join(hbase, 'h_pca.pkl')
            h_pco = joblib.load(pcaf)
            # Load npy whpa prediction
            prediction = np.load(os.path.join(hbase, 'roots_whpa', f'{root}.npy'))
            # Transform and split
            h_pco.pca_test_transformation(prediction, test_root=[root])
            nho = h_pco.ncomp
            h_pc_training, h_pc_prediction = h_pco.pca_refresh(nho)
            # Plot
            fig_file = os.path.join(hbase, 'roots_whpa', f'{root}_pca_scores.png')
            if scores:
                pca_scores(training=h_pc_training,
                           prediction=h_pc_prediction,
                           n_comp=nho,
                           labels=False,
                           fig_file=fig_file)
            # Explained variance plots
            if exvar:
                fig_file = os.path.join(hbase, 'roots_whpa', f'{root}_pca_exvar.png')
                explained_variance(h_pco.operator, n_comp=h_pco.ncomp, thr=.85, fig_file=fig_file)

    @staticmethod
    def cca_vision(root, folders=None):
        """
        Loads CCA pickles and plots components for all folders
        :param root:
        :param folders:
        :return:
        """

        if isinstance(root, (list, tuple)):
            if len(root) > 1:
                print('Input error')
                return
            else:
                root = root[0]

        subdir = os.path.join(MySetup.Directories.forecasts_dir, root)

        if folders is None:
            listme = os.listdir(subdir)
            folders = list(filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))
        else:
            if not isinstance(folders, (list, tuple)):
                folders = [folders]
            else:
                pass

        base_dir = os.path.join(MySetup.Directories.forecasts_dir, 'base')

        for f in folders:
            res_dir = os.path.join(subdir, f, 'obj')
            # Load objects
            f_names = list(map(lambda fn: os.path.join(res_dir, fn + '.pkl'), ['cca', 'd_pca']))
            cca_operator, d_pco = list(map(joblib.load, f_names))
            h_pco = joblib.load(os.path.join(base_dir, 'h_pca.pkl'))

            h_pred = np.load(os.path.join(base_dir, 'roots_whpa', f'{root}.npy'))

            # Inspect transformation between physical and PC space
            dnc0 = d_pco.ncomp
            hnc0 = h_pco.ncomp

            # Cut desired number of PC components
            d_pc_training, d_pc_prediction = d_pco.pca_refresh(dnc0)
            h_pco.pca_test_transformation(h_pred, test_root=[root])
            h_pc_training, h_pc_prediction = h_pco.pca_refresh(hnc0)

            # CCA plots
            d_cca_training, h_cca_training = cca_operator.transform(d_pc_training, h_pc_training)
            d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T

            cca_coefficient = np.corrcoef(d_cca_training, h_cca_training, ).diagonal(offset=cca_operator.n_components)

            cca_plot(cca_operator, d_cca_training, h_cca_training, d_pc_prediction, h_pc_prediction,
                     sdir=os.path.join(os.path.dirname(res_dir), 'cca'))

            sns.lineplot(data=cca_coefficient)
            plt.grid(alpha=.2, linewidth=.5)
            plt.title('Decrease of CCA correlation coefficient with component number')
            plt.ylabel('Correlation coefficient')
            plt.xlabel('Component number')
            plt.savefig(os.path.join(os.path.dirname(res_dir), 'cca', 'coefs.png'), dpi=300, transparent=True)
            # plt.show()

    @staticmethod
    def plot_whpa(root=None):
        """
        Loads target pickle and plots all training WHPA
        :param root:
        :return:
        """

        if isinstance(root, (list, tuple)):
            if len(root) > 1:
                print('Input error')
                return
            else:
                root = root[0]

        base_dir = os.path.join(MySetup.Directories.forecasts_dir, 'base')
        x_lim, y_lim, grf = MySetup.Focus.x_range, MySetup.Focus.y_range, MySetup.Focus.cell_dim
        mplot = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf)

        fobj = os.path.join(MySetup.Directories.forecasts_dir, 'base', 'h_pca.pkl')
        h = joblib.load(fobj)
        h_training = h.training_physical.reshape(h.training_shape)

        mplot.whp(h_training)

        if root is not None:
            h_pred = np.load(os.path.join(base_dir, 'roots_whpa', f'{root}.npy'))
            mplot.whp(h=h_pred, colors='red', lw=1, alpha=1,
                      fig_file=os.path.join(MySetup.Directories.forecasts_dir, root, 'whpa_training.png'))

    @staticmethod
    def plot_pc_ba(root, data=False, target=False):
        """

        :param root:
        :param data:
        :param target:
        :return:
        """

        if isinstance(root, (list, tuple)):
            if len(root) > 1:
                print('Input error')
                return
            else:
                root = root[0]

        x_lim, y_lim, grf = MySetup.Focus.x_range, MySetup.Focus.y_range, MySetup.Focus.cell_dim
        mplot = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf)

        base_dir = os.path.join(MySetup.Directories.forecasts_dir, 'base')
        if target:
            fobj = os.path.join(base_dir, 'h_pca.pkl')
            h_pco = joblib.load(fobj)
            hnc0 = h_pco.ncomp
            # mplot.h_pca_inverse_plot(h_pco, hnc0, training=True, fig_dir=os.path.join(base_dir, 'control'))

            h_pred = np.load(os.path.join(base_dir, 'roots_whpa', f'{root}.npy'))
            # Cut desired number of PC components
            h_pco.pca_test_transformation(h_pred, test_root=[root])
            h_pco.pca_refresh(hnc0)
            mplot.h_pca_inverse_plot(h_pco, hnc0, training=False, fig_dir=os.path.join(base_dir, 'control'))

        # d
        if data:
            subdir = os.path.join(MySetup.Directories.forecasts_dir, root)
            listme = os.listdir(subdir)
            folders = list(filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))

            for f in folders:
                res_dir = os.path.join(subdir, f, 'obj')
                # Load objects
                d_pco = joblib.load(os.path.join(res_dir, 'd_pca.pkl'))
                dnc0 = d_pco.ncomp
                d_pco.pca_refresh(dnc0)
                setattr(d_pco, 'test_root', [root])
                mplot.d_pca_inverse_plot(d_pco, dnc0, training=True,
                                         fig_dir=os.path.join(os.path.dirname(res_dir), 'pca'))
                mplot.d_pca_inverse_plot(d_pco, dnc0, training=False,
                                         fig_dir=os.path.join(os.path.dirname(res_dir), 'pca'))
