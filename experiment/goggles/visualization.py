#  Copyright (c) 2020. Robin Thibaut, Ghent University

from os.path import join as jp

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline

from experiment.base.inventory import MySetup
from experiment.toolbox import filesio

plt.style.use('dark_background')


def d_pca_inverse_plot(v, e, pca_o, vn, fig_file=None, show=False):
    """
    Plot used to compare the reproduction of the original physical space after PCA transformation
    :param v: Original, untransformed data array
    :param e: Sample number on which the test is performed
    :param pca_o: data PCA operator
    :param vn: Number of components to inverse-transform the data
    :return:
    """
    v_pc = pca_o.transform(v)
    v_pred = np.dot(v_pc[e, :vn], pca_o.components_[:vn, :]) + pca_o.mean_
    plt.plot(v[e], 'r', alpha=.8)
    plt.plot(v_pred, 'c', alpha=.8)
    if fig_file is not None:
        plt.savefig(fig_file, dpi=300)
        plt.close()
    if show:
        plt.show()
        plt.close()


def explained_variance(pca, n_comp=0, xfs=2, fig_file=None, show=False):
    """
    PCA explained variance plot
    :param pca: PCA operator
    :param n_comp: Number of components to display
    :param xfs: X-axis fontsize
    :param fig_file:
    :param show:
    :return:
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


def pca_scores(training, prediction, n_comp, fig_file=None, show=False):
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
    plt.grid(alpha=0.2)
    ut = n_comp
    plt.xticks(np.arange(0, ut, 5), fontsize=8)
    plt.plot(training.T[:ut], '+w', markersize=.5, alpha=0.2)  # Plot all training scores
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=.15, reverse=True)
    sns.kdeplot(np.arange(1, ut+1), training.T[:ut][:, 1], cmap=cmap, n_levels=60, shade=True, vertical=True)
    plt.xlim(0.5, ut)  # Define x lim
    for sample_n in range(len(prediction)):
        pc_obs = prediction[sample_n]
        plt.plot(pc_obs.T[:ut],  # Plot observations scores
                 'o', markersize=2.5, markeredgecolor='k', markeredgewidth=.4, alpha=.8,
                 label=str(sample_n))

        xnew = np.linspace(1, ut, 200)
        spl = make_interp_spline(np.arange(1, ut+1), pc_obs.T[:ut], k=3)  # type: BSpline
        power_smooth = spl(xnew)
        plt.plot(xnew-1, power_smooth, 'y', linewidth=.3, alpha=.5)

    plt.title('PCA scores')
    plt.xlabel('Component id')
    plt.ylabel('Component scores')
    plt.tick_params(labelsize=6)

    if fig_file:
        plt.savefig(fig_file, dpi=300)
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
            d_obs = d_pc_prediction[sample_n]
            h_obs = h_pc_prediction[sample_n]
            d_cca_prediction, h_cca_prediction = cca_operator.transform(d_obs.reshape(1, -1),
                                                                        h_obs.reshape(1, -1))
            d_cca_prediction, h_cca_prediction = d_cca_prediction.T, h_cca_prediction.T

            cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
            g = sns.jointplot(d[comp_n], h[comp_n],
                              cmap=cmap, n_levels=60, shade=True,
                              kind='kde')
            g.plot_joint(plt.scatter, c='w', marker='+', s=2, alpha=.7)
            plt.plot(d_cca_prediction[comp_n], h_cca_prediction[comp_n],
                     'wo', markersize=4.5, markeredgecolor='k', alpha=.7,
                     label='{}'.format(sample_n))
        # plt.grid('w', linewidth=.3, alpha=.4)
        # plt.tick_params(labelsize=8)
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(round(cca_coefficient[i], 4))
        if sdir:
            plt.savefig(jp(sdir, 'cca{}.png'.format(i)), bbox_inches='tight', dpi=300)
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
        self.cols = ['w', 'g', 'r', 'c', 'm', 'y']

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
        v = np.array([c0.allsegs[0][0] for c0 in c0s])
        return v

    def curves(self, tc, highlight=None, sdir=None, show=False):
        """
        Shows every breakthrough curve stacked on a plot.
        :param tc: Curves with shape (n_sim, n_wels, n_time_steps)
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
            plt.savefig(jp(sdir, f'{title}.png'), dpi=300)
            plt.close()
        if show:
            plt.show()
            plt.close()

    def curves_i(self, tc, highlight=None, sdir=None, show=False):
        """
        Shows every breakthrough individually for each observation point.
        Will produce n_well figures of n_sim curves each.
        :param tc: Curves with shape (n_sim, n_wells, n_time_steps)
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
                plt.savefig(jp(sdir, f'{title}_{t + 1}.png'), dpi=300)
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
        Produces the WHPA plot, that is the zero-contour of the signed distance array.
        It assumes that well information can be loaded from pw.npy and iw.npy.
        I should change this.
        :param show_wells: whether to plot well coordinates or not
        :param cmap: colormap for the background array
        :param vmax: max value to plot for the background array
        :param vmin: max value to plot for the background array
        :param bkg_field_array: 2D array whose values will be plotted on the grid
        :param h: Array containing grids of values whose 0 contour will be computed and plotted
        :param alpha: opacity of the 0 contour lines
        :param lw: Line width
        :param colors: Line color
        :param fig_file:
        :param show:
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
            pwl = wbd['pumping0']['coordinates']
            plt.plot(pwl[0], pwl[1], 'wo', label='pw')
            for n, i in enumerate(wbd):
                if 'pumping' not in i:
                    plt.plot(wbd[i]['coordinates'][0], wbd[i]['coordinates'][1],
                             'o', markersize=4, markeredgecolor='k', markeredgewidth=.5,
                             label='{}'.format(n))
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
            plt.savefig(fig_file, bbox_inches='tight', dpi=300)
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

        self.whp(h=forecasts, show_wells=show_wells, bkg_field_array=bkg_field_array, title=title)
        # Plot true h
        plt.contour(self.x, self.y, h_true, [0], colors='red', linewidths=1, alpha=.9)
        # Plot true h predicted
        if h_pred is not None:
            plt.contour(self.x, self.y, h_pred, [0], colors='cyan', linewidths=1, alpha=.9)
        if fig_file:
            plt.savefig(fig_file, bbox_inches='tight', dpi=300)
            plt.close()
        if show:
            plt.show()
            plt.close()

    def h_pca_inverse_plot(self, pca_o, e, vn, fig_file=None, show=False):
        """
        Plot used to compare the reproduction of the original physical space after PCA transformation
        :param e: Sample number on which the test is performed
        :param pca_o: signed distance PCA operator
        :param vn: Number of components to inverse-transform.
        :return:
        """
        shape = pca_o.shape
        v_pc = pca_o.training_pc
        v_pred = (np.dot(v_pc[e, :vn], pca_o.operator.components_[:vn, :]) + pca_o.operator.mean_)
        self.whp(h=v_pred.reshape(1, shape[1], shape[2]), colors='cyan', alpha=.8, lw=1, show=False)
        self.whp(h=pca_o.training_physical[e].reshape(1, shape[1], shape[2]), colors='red', alpha=1, lw=1, show=False)

        if fig_file is not None:
            plt.savefig(fig_file, dpi=300)
            plt.close()
        if show:
            plt.show()
            plt.close()

    def pca_inverse_compare(self, pco_d, pco_h, nd=None, nh=None, show=False):
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

            d_pca_inverse_plot(pco_d.training_physical,
                               n_compare,
                               pco_d.operator,
                               nd,
                               fig_file=''.join((fig_file, '_d')), show=show)

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
