#  Copyright (c) 2020. Robin Thibaut, Ghent University
import itertools
import os
from os.path import join as jp
import warnings
import string

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline

from experiment.base.inventory import MySetup
from experiment.toolbox import filesio
from sklearn.preprocessing import PowerTransformer

ftype = 'png'


def my_alphabet(az):
    alphabet = string.ascii_uppercase
    extended_alphabet = [''.join(i) for i in list(itertools.permutations(alphabet, 2))]

    if az <= 25:
        sub = alphabet[az]
    else:
        j = az - 26
        sub = extended_alphabet[j]

    return sub


def proxy_legend(legend1=None,
                 colors: list = None,
                 labels: list = None,
                 loc: int = 4,
                 marker: str = '-',
                 pec: list = None,
                 fz: float = 11,
                 fig_file: str = None):
    """
    Add a second legend to a figure @ bottom right (loc=4)
    https://stackoverflow.com/questions/12761806/matplotlib-2-different-legends-on-same-graph
    :param legend1: First legend instance from the figure
    :param colors: List of colors
    :param labels: List of labels
    :param loc: Position of the legend
    :param marker: Points 'o' or line '-'
    :param pec: List of point edge color, e.g. [None, 'k']
    :param fz: Fontsize
    :param fig_file: Path to figure file
    :return:
    """
    if colors is None:
        colors = ['w']
    if labels is None:
        labels = []
    if pec is None:
        pec = [None for _ in range(len(colors))]

    proxys = [plt.plot([], marker, color=c, markeredgecolor=pec[i]) for i, c in enumerate(colors)]
    plt.legend([p[0] for p in proxys], labels, loc=loc, fontsize=fz)

    if legend1:
        plt.gca().add_artist(legend1)

    if fig_file:
        filesio.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()


def proxy_annotate(annotation: list, loc: int = 1, fz: float = 11):
    """
    Places annotation (or title) within the figure box
    :param annotation: Must be a list of labels even of it only contains one label. Savvy ?
    :param fz: Fontsize
    :param loc: Location (default 1 = upper right corner)
    :return:
    """

    legend_a = plt.legend(plt.plot([], linestyle=None, color='w', markeredgecolor=None),
                          annotation,
                          handlelength=0, handletextpad=0, fancybox=True, loc=loc, fontsize=fz)

    return legend_a


def explained_variance(pca,
                       n_comp: int = 0,
                       thr: float = 1.,
                       annotation: list = None,
                       fig_file: str = None,
                       show: bool = False):
    """
    PCA explained variance plot
    :param pca: PCA operator
    :param n_comp: Number of components to display
    :param thr: float: Threshold
    :param fig_file:
    :param show:
    :return:
    """
    plt.grid(alpha=0.1)
    if not n_comp:
        n_comp = pca.n_components_

    # plt.xticks(np.arange(n_comp), fontsize=xfs)
    # Index where explained variance is below threshold:
    ny = len(np.where(np.cumsum(pca.explained_variance_ratio_) < thr)[0])
    # Explained variance vector:
    cum = np.cumsum(pca.explained_variance_ratio_[:n_comp]) * 100
    # Tricky y-ticks
    yticks = np.append(cum[:ny], cum[-1])
    plt.yticks(yticks, fontsize=8.5)
    # bars for aesthetics
    plt.bar(np.arange(n_comp),
            np.cumsum(pca.explained_variance_ratio_[:n_comp]) * 100,
            color='m', alpha=.1)
    # line for aesthetics
    plt.plot(np.arange(n_comp),
             np.cumsum(pca.explained_variance_ratio_[:n_comp]) * 100,
             '-o', linewidth=.5, markersize=1.5, alpha=.8)

    plt.xlabel('PC number', fontsize=11)
    plt.ylabel('Cumulative explained variance (%)', fontsize=11)
    legend_a = proxy_annotate(annotation=annotation, loc=2)
    plt.gca().add_artist(legend_a)
    plt.legend(fontsize=14)

    if fig_file:
        filesio.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, dpi=300, transparent=True)
        plt.close()
    if show:
        plt.show()
        plt.close()


def pca_scores(training,
               prediction,
               n_comp: int,
               annotation: list,
               fig_file: str = None,
               labels: bool = True,
               show: bool = False):
    """
    PCA scores plot, displays scores of observations above those of training.
    :param labels:
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
    plt.xticks(np.arange(0, ut, 5), fontsize=11)
    # Plot all training scores
    plt.plot(training.T[:ut], 'ob', markersize=3, alpha=0.05)
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
        plt.plot(xnew - 1,
                 power_smooth,
                 'red',
                 linewidth=1.2,
                 alpha=.9)
        # plt.plot(xnew - 1, power_smooth, 'y', linewidth=.3, alpha=.5)

        plt.plot(pc_obs.T[:ut],  # Plot observations scores
                 'ro',
                 markersize=3,
                 markeredgecolor='k',
                 markeredgewidth=.4,
                 alpha=.8,
                 label=str(sample_n))
        # plt.plot(pc_obs.T[:ut],  # Plot observations scores
        #          'o', markersize=2.5, markeredgecolor='k', markeredgewidth=.4, alpha=.8,
        #          label=str(sample_n))

    if labels:
        plt.title('Principal Components of training and test dataset')
        plt.xlabel('PC number')
        plt.ylabel('PC')
    plt.tick_params(labelsize=11)
    # Add legend
    # Add title inside the box
    legend_a = proxy_annotate(annotation=annotation, loc=2, fz=14)
    proxy_legend(legend1=legend_a,
                 colors=['blue', 'red'],
                 labels=['Training', 'Test'],
                 marker='o')

    if fig_file:
        filesio.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, dpi=300, transparent=True)
        plt.close()
    if show:
        plt.show()
        plt.close()


def cca_plot(cca_operator,
             d,
             h,
             d_pc_prediction,
             h_pc_prediction,
             sdir=None,
             show=False):
    """
    CCA plots.
    Receives d, h PC components to be predicted, transforms them in CCA space and adds it to the plots.
    :param cca_operator: CCA operator
    :param d: d CCA scores
    :param h: h CCA scores
    :param d_pc_prediction: d test PC scores
    :param h_pc_prediction: h test PC scores
    :param sdir: str:
    :param show: bool:
    :return:
    """

    cca_coefficient = np.corrcoef(d, h).diagonal(offset=cca_operator.n_components)  # Gets correlation coefficient
    #
    # post_obj = joblib.load(jp(os.path.dirname(sdir), 'obj', 'post.pkl'))
    # h_samples = post_obj.random_sample()
    #

    # CCA plots for each observation:
    for i in range(cca_operator.n_components):
        comp_n = i
        # plt.plot(d[comp_n], h[comp_n], 'w+', markersize=3, markerfacecolor='w', alpha=.25)
        for sample_n in range(len(d_pc_prediction)):  # For each 'observation'
            # Extract from sample
            d_obs = d_pc_prediction[sample_n]
            h_obs = h_pc_prediction[sample_n]
            # Transform to CCA space and transpose
            d_cca_prediction, h_cca_prediction = cca_operator.transform(d_obs.reshape(1, -1),
                                                                        h_obs.reshape(1, -1))
            # d_cca_prediction, h_cca_prediction = d_cca_prediction.T, h_cca_prediction.T

            # %%
            h2 = h.copy()
            d2 = d.copy()
            tfm1 = PowerTransformer(method='yeo-johnson', standardize=True)
            h = tfm1.fit_transform(h2.T)
            h = h.T
            h_cca_prediction = tfm1.transform(h_cca_prediction)
            h_cca_prediction = h_cca_prediction.T

            tfm2 = PowerTransformer(method='yeo-johnson', standardize=True)
            d = tfm2.fit_transform(d2.T)
            d = d.T
            d_cca_prediction = tfm2.transform(d_cca_prediction)
            d_cca_prediction = d_cca_prediction.T

            # %%
            # Choose beautiful color map
            # cube_helix very nice for dark mode
            # light = 0.95 is beautiful for reverse = True
            # cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=False)
            cmap = sns.color_palette("Blues", as_cmap=True)
            # Seaborn 'joinplot' between d & h training CCA scores
            g = sns.jointplot(d[comp_n], h[comp_n],
                              cmap=cmap, n_levels=80, shade=True,
                              kind='kde')
            g.plot_joint(plt.scatter, c='w', marker='o', s=2, alpha=.7)
            # add 'arrows' at observation location - tricky part!
            # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.FancyArrow.html
            g.ax_marg_x.arrow(d_cca_prediction[comp_n], 0, 0, .1, color='r', head_width=0, head_length=0, lw=2)
            g.ax_marg_y.arrow(0, h_cca_prediction[comp_n], .1, 0, color='r', head_width=0, head_length=0, lw=2)
            # Plot prediction (d, h) in canonical space
            plt.plot(d_cca_prediction[comp_n], h_cca_prediction[comp_n],
                     'ro', markersize=4.5, markeredgecolor='k', alpha=1,
                     label=f'{sample_n}')
            # Plot predicted canonical variate mean
            # plt.plot(np.ones(post_obj.n_posts)*d_cca_prediction[comp_n], h_samples[comp_n],
            #          'bo', markersize=4.5, markeredgecolor='w', alpha=.7,
            #          label='{}'.format(sample_n))

        # plt.grid('w', linewidth=.3, alpha=.4)
        # plt.tick_params(labelsize=8)
        plt.xlabel('$d^{c}$', fontsize=14)
        plt.ylabel('$h^{c}$', fontsize=14)
        plt.subplots_adjust(top=0.9)
        plt.tick_params(labelsize=14)
        # g.fig.suptitle(f'Pair {i + 1} - R = {round(cca_coefficient[i], 3)}', fontsize=11)
        # Put title inside box

        subtitle = my_alphabet(i)

        # Add title inside the box
        an = [f'{subtitle}. Pair {i + 1} - R = {round(cca_coefficient[i], 3)}']
        legend_a = proxy_annotate(annotation=an, loc=2, fz=14)

        proxy_legend(legend1=legend_a,
                     colors=['white', 'red'],
                     labels=['Training', 'Test'],
                     marker='o',
                     pec=['k', 'k'])

        if sdir:
            filesio.dirmaker(sdir)
            plt.savefig(jp(sdir, 'cca{}.png'.format(i)), bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
        if show:
            plt.show()
            plt.close()


class Plot:

    def __init__(self,
                 x_lim=None,
                 y_lim=None,
                 grf=None,
                 well_comb=None):

        md = MySetup.Directories()
        focus = MySetup.Focus()
        self.wells = MySetup.Wells()

        if well_comb is not None:
            self.wells.combination = well_comb

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

        wells_id = list(self.wells.wells_data.keys())
        self.cols = [self.wells.wells_data[w]['color'] for w in wells_id if 'pumping' not in w]

    def contours_vertices(self,
                          arrays,
                          c=0):
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

    def curves(self,
               tc,
               highlight=None,
               ghost=False,
               sdir=None,
               labelsize=12,
               factor=1,
               xlabel=None,
               ylabel=None,
               title='curves',
               show=False):
        """
        Shows every breakthrough curve stacked on a plot.
        :param ylabel:
        :param xlabel:
        :param factor:
        :param labelsize:
        :param tc: Curves with shape (n_sim, n_wells, n_time_steps)
        :param highlight: list: List of indices of curves to highlight in the plot
        :param ghost: bool: Flag to only display highlighted curves.
        :param sdir: Directory in which to save figure
        :param title: str: Title
        :param show: Whether to show or not
        """
        if highlight is None:
            highlight = []
        n_sim, n_wells, nts = tc.shape
        for i in range(n_sim):
            for t in range(n_wells):
                if i in highlight:
                    plt.plot(tc[i][t] * factor, color=self.cols[t], linewidth=2, alpha=1)
                elif not ghost:
                    plt.plot(tc[i][t] * factor, color=self.cols[t], linewidth=.2, alpha=0.5)

        plt.grid(linewidth=.3, alpha=.4)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tick_params(labelsize=labelsize)
        if sdir:
            filesio.dirmaker(sdir)
            plt.savefig(jp(sdir, f'{title}.png'), dpi=300, transparent=True)
            plt.close()
        if show:
            plt.show()
            plt.close()

    def curves_i(self,
                 tc,
                 highlight=None,
                 labelsize=12,
                 factor=1,
                 xlabel=None,
                 ylabel=None,
                 sdir=None,
                 show=False):
        """
        Shows every breakthrough individually for each observation point.
        Will produce n_well figures of n_sim curves each.
        :param labelsize:
        :param factor:
        :param xlabel:
        :param ylabel:
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
                    plt.plot(tc[i][t] * factor, color='k', linewidth=2, alpha=1)
                else:
                    plt.plot(tc[i][t] * factor, color=self.cols[t], linewidth=.2, alpha=0.5)
            colors = [self.cols[t], 'k']
            plt.grid(linewidth=.3, alpha=.4)
            plt.tick_params(labelsize=labelsize)
            # plt.title(f'Well {t + 1}')

            alphabet = string.ascii_uppercase
            legend_a = proxy_annotate([f'{alphabet[t]}. Well {t + 1}'], fz=12, loc=2)

            labels = ['Training', 'Test']
            proxy_legend(legend1=legend_a, colors=colors, labels=labels, loc=1)

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if sdir:
                filesio.dirmaker(sdir)
                plt.savefig(jp(sdir, f'{title}_{t + 1}.png'), dpi=300, transparent=True)
                plt.close()
            if show:
                plt.show()
                plt.close()

    def plot_wells(self,
                   well_ids=None,
                   markersize: float = 4.):

        if well_ids is None:
            comb = [0] + list(self.wells.combination)
        else:
            comb = well_ids
        # comb = [0] + list(self.wells.combination)
        # comb = [0] + list(self.wells.combination)
        keys = [list(self.wells.wells_data.keys())[i] for i in comb]
        wbd = {k: self.wells.wells_data[k] for k in keys if k in self.wells.wells_data}
        s = 0
        for i in wbd:
            n = comb[s]
            if n == 0:
                label = 'pw'
            else:
                label = f'{n}'
            if n in comb:
                plt.plot(wbd[i]['coordinates'][0],
                         wbd[i]['coordinates'][1],
                         f'{wbd[i]["color"]}o',
                         markersize=markersize,
                         markeredgecolor='k',
                         markeredgewidth=.5,
                         label=label)
            s += 1

    def whp(self,
            h=None,
            alpha=0.4,
            lw=.5,
            bkg_field_array=None,
            vmin=None,
            vmax=None,
            x_lim=None,
            y_lim=None,
            xlabel=None,
            ylabel=None,
            cb_title=None,
            labelsize=5,
            cmap='coolwarm',
            colors='white',
            show_wells=False,
            well_ids=None,
            title=None,
            fig_file=None,
            show=False):
        """
        Produces the WHPA plot, i.e. the zero-contour of the signed distance array.

        :param xlabel:
        :param ylabel:
        :param cb_title:
        :param well_ids:
        :param labelsize: Label size
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
            cb = plt.colorbar()
            cb.ax.set_title(cb_title)

        contour = None
        # Plot results
        if h is None:
            h = []

        for z in h:  # h is the n square WHPA matrix
            contour = plt.contour(self.x, self.y, z, [0], colors=colors, linewidths=lw, alpha=alpha)
        plt.grid(color='c', linestyle='-', linewidth=.5, alpha=.2)

        # Plot wells
        well_legend = None
        if show_wells:
            self.plot_wells(well_ids=well_ids, markersize=7)
            well_legend = plt.legend(fontsize=11)

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

        plt.xlabel(xlabel, fontsize=labelsize)
        plt.ylabel(ylabel, fontsize=labelsize)

        # Tick size
        plt.tick_params(labelsize=labelsize)

        if fig_file:
            filesio.dirmaker(os.path.dirname(fig_file))
            plt.savefig(fig_file, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
        if show:
            plt.show()
            plt.close()

        return contour, well_legend

    def whp_prediction(self,
                       forecasts,
                       h_true,
                       h_pred=None,
                       x_lim=None,
                       y_lim=None,
                       label=None,
                       bkg_field_array=None,
                       fig_file=None,
                       show_wells=False,
                       well_ids=None,
                       title=None,
                       show=False):
        warnings.warn('Depecrated funcion')
        # Plot n forecasts sampled
        self.whp(h=forecasts,
                 x_lim=x_lim,
                 y_lim=y_lim,
                 show_wells=show_wells,
                 well_ids=well_ids,
                 bkg_field_array=bkg_field_array,
                 title=title)

        # Plot true h
        plt.contour(self.x, self.y, h_true, [0], colors='red', linewidths=1, alpha=.9, label=label)

        # Plot 'true' h predicted
        if h_pred is not None:
            plt.contour(self.x, self.y, h_pred, [0], colors='blue', linewidths=1, alpha=.9)
        if fig_file:
            filesio.dirmaker(os.path.dirname(fig_file))
            plt.savefig(fig_file, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
        if show:
            plt.show()
            plt.close()

    @staticmethod
    def d_pca_inverse_plot(pca_o,
                           factor: float = 1.,
                           xlabel: str = None,
                           ylabel: str = None,
                           labelsize: float = 11.,
                           training=True,
                           fig_dir=None,
                           show=False):
        """

        Plot used to compare the reproduction of the original physical space after PCA transformation.

        :param xlabel:
        :param ylabel:
        :param labelsize:
        :param factor:
        :param pca_o: data PCA operator
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

            # v_pred = np.dot(v_pc[i, :vn], pca_o.operator.components_[:vn, :]) + pca_o.operator.mean_
            # The trick is to use [0]
            v_pred = pca_o.custom_inverse_transform(v_pc)[0]

            if training:
                to_plot = np.copy(pca_o.training_physical[i])
            else:
                to_plot = np.copy(pca_o.predict_physical[i])

            plt.plot(to_plot * factor, 'r', alpha=.8)
            plt.plot(v_pred * factor, 'b', alpha=.8)
            # Add title inside the box
            an = ['A']
            legend_a = proxy_annotate(annotation=an, loc=2, fz=14)
            proxy_legend(legend1=legend_a,
                         colors=['red', 'blue'],
                         labels=['Physical', 'Back transformed'],
                         marker='-')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.tick_params(labelsize=labelsize)

            if fig_dir is not None:
                filesio.dirmaker(fig_dir)
                plt.savefig(jp(fig_dir, f'{r}_d.png'), dpi=300, transparent=True)
                plt.close()
            if show:
                plt.show()
                plt.close()

    def h_pca_inverse_plot(self,
                           pca_o,
                           vn,
                           training=True,
                           fig_dir=None,
                           show=False):
        """

        Plot used to compare the reproduction of the original physical space after PCA transformation

        :param pca_o: signed distance PCA operator
        :param vn: Number of components to keep while inverse-transforming the data
        :param training: bool:
        :param fig_dir: str:
        :param show: bool:
        :return:
        """

        shape = pca_o.training_shape

        if training:
            v_pc = pca_o.training_pc
            roots = pca_o.roots
        else:
            v_pc = pca_o.predict_pc
            roots = pca_o.test_root

        for i, r in enumerate(roots):

            if training:
                h_to_plot = np.copy(pca_o.training_physical[i].reshape(1, shape[1], shape[2]))
            else:
                h_to_plot = np.copy(pca_o.predict_physical[i].reshape(1, shape[1], shape[2]))
            self.whp(h=h_to_plot,
                     colors='red', alpha=1, lw=2)
            # v_pred = (np.dot(v_pc[i, :vn], pca_o.operator.components_[:vn, :]) + pca_o.operator.mean_)
            v_pred = pca_o.custom_inverse_transform(v_pc)

            self.whp(h=v_pred.reshape(1, shape[1], shape[2]),
                     colors='blue',
                     alpha=1,
                     lw=2,
                     labelsize=11,
                     xlabel='X(m)',
                     ylabel='Y(m)',
                     x_lim=[850, 1100],
                     y_lim=[350, 650])

            # Add title inside the box
            an = ['B']

            legend_a = proxy_annotate(annotation=an,
                                      loc=2,
                                      fz=14)

            proxy_legend(legend1=legend_a,
                         colors=['red', 'blue'],
                         labels=['Physical', 'Back transformed'],
                         marker='-')

            if fig_dir is not None:
                filesio.dirmaker(fig_dir)
                plt.savefig(jp(fig_dir, f'{r}_h.png'), dpi=300, transparent=True)
                plt.close()

            if show:
                plt.show()
                plt.close()

    def check_root(self, root):
        """
        Plots raw data of folder 'root'
        :param root:
        :return:
        """
        bkt, whpa, _ = filesio.data_loader(roots=root)
        whpa = whpa.squeeze()
        # self.curves_i(bkt, show=True)  # This function will not work

        plt.plot(whpa[:, 0], whpa[:, 1], 'wo')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.show()

    def plot_results(self,
                     root,
                     folder):
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

        # Plot parameters for predictor
        xlabel = 'Observation index number'
        ylabel = 'Concentration ($g/m^{3})$'
        factor = 1000
        labelsize = 11

        self.curves(tc=tc,
                    sdir=sdir,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    factor=factor,
                    labelsize=labelsize,
                    highlight=[len(tc) - 1])

        self.curves(tc=tc,
                    sdir=sdir,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    factor=factor,
                    labelsize=labelsize,
                    highlight=[len(tc) - 1],
                    ghost=True,
                    title='curves_ghost')

        self.curves_i(tc=tc,
                      xlabel=xlabel,
                      ylabel=ylabel,
                      factor=factor,
                      labelsize=labelsize,
                      sdir=sdir,
                      highlight=[len(tc) - 1])

        # WHP - h test + training
        fig_dir = jp(hbase, 'roots_whpa')
        ff = jp(fig_dir, f'{root}.png')  # figure name
        h = np.load(jp(fig_dir, f'{root}.npy')).reshape(h_pco.obs_shape)
        h_training = h_pco.training_physical.reshape(h_pco.training_shape)
        # Plots target training + prediction
        self.whp(h_training, colors='blue', alpha=.2)
        self.whp(h, colors='r', lw=2, alpha=.8, xlabel='X(m)', ylabel='Y(m)', labelsize=11)
        colors = ['blue', 'red']
        labels = ['Training', 'Test']
        proxy_legend(colors=colors, labels=labels, fig_file=ff)

        # WHPs
        ff = jp(md,
                'uq',
                f'cca_{cca_operator.n_components}.png')
        h_training = h_pco.training_physical.reshape(h_pco.training_shape)
        post_obj = joblib.load(jp(md, 'obj', 'post.pkl'))
        forecast_posterior = post_obj.bel_predict(pca_d=d_pco,
                                                  pca_h=h_pco,
                                                  cca_obj=cca_operator,
                                                  n_posts=MySetup.Forecast.n_posts,
                                                  add_comp=False)

        # I display here the prior h behind the forecasts sampled from the posterior.
        well_ids = [0] + list(map(int, list(folder)))
        labels = ['Training', 'Samples', 'True test']
        colors = ['darkblue', 'darkred', 'k']

        # Training
        _, well_legend = self.whp(h_training,
                                  alpha=.5,
                                  lw=.5,
                                  colors=colors[0],
                                  show_wells=True,
                                  well_ids=well_ids,
                                  show=False)

        # Samples
        self.whp(forecast_posterior,
                 colors=colors[1],
                 lw=1,
                 alpha=1,
                 show=False)

        # True test
        self.whp(h,
                 colors=colors[2],
                 lw=1,
                 alpha=1,
                 x_lim=[800, 1200],
                 xlabel='X(m)',
                 ylabel='Y(m)',
                 labelsize=11)

        # Tricky operation to add a second legend:
        proxy_legend(legend1=well_legend,
                     colors=colors,
                     labels=labels,
                     fig_file=ff)

    def plot_K_field(self, root):
        # HK field
        matrix = np.load(jp(MySetup.Directories.hydro_res_dir, root, 'hk0.npy'))
        grid_dim = MySetup.GridDimensions
        extent = (grid_dim.xo, grid_dim.x_lim, grid_dim.yo, grid_dim.y_lim)
        plt.imshow(np.log10(matrix), cmap='coolwarm', extent=extent)
        self.plot_wells(markersize=1)
        plt.colorbar()
        plt.savefig(jp(MySetup.Directories.forecasts_dir, root, 'k_field.png'),
                    bbox_inches='tight',
                    dpi=300,
                    transparent=True)
        plt.close()

    @staticmethod
    def pca_vision(root,
                   d=True,
                   h=False,
                   scores=True,
                   exvar=True,
                   labels=False,
                   folders=None):
        """
        Loads PCA pickles and plot scores for all folders
        :param labels:
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

        i = 0

        if d:
            for f in folders:
                dfig = os.path.join(subdir, f, 'pca')
                # For d only
                pcaf = os.path.join(subdir, f, 'obj', 'd_pca.pkl')
                d_pco = joblib.load(pcaf)
                fig_file = os.path.join(dfig, 'd_scores.png')
                if scores:
                    annotation = [my_alphabet(i)]

                    pca_scores(training=d_pco.training_pc,
                               prediction=d_pco.predict_pc,
                               n_comp=d_pco.n_pc_cut,
                               annotation=annotation,
                               labels=labels,
                               fig_file=fig_file)
                # Explained variance plots
                if exvar:
                    i += 1
                    annotation = [my_alphabet(i)]

                    fig_file = os.path.join(dfig, 'd_exvar.png')
                    explained_variance(d_pco.operator,
                                       n_comp=d_pco.n_pc_cut,
                                       thr=.8,
                                       annotation=annotation,
                                       fig_file=fig_file)
        if h:
            hbase = os.path.join(MySetup.Directories.forecasts_dir, 'base')
            # Load h pickle
            pcaf = os.path.join(hbase, 'h_pca.pkl')
            h_pco = joblib.load(pcaf)
            # Load npy whpa prediction
            prediction = np.load(os.path.join(hbase, 'roots_whpa', f'{root}.npy'))
            # Transform and split
            h_pco.pca_test_fit_transform(prediction, test_root=[root])
            nho = h_pco.n_pc_cut
            h_pc_training, h_pc_prediction = h_pco.pca_refresh(nho)
            # Plot
            fig_file = os.path.join(hbase, 'roots_whpa', f'{root}_pca_scores.png')
            if scores:
                i += 1
                annotation = [my_alphabet(i)]

                pca_scores(training=h_pc_training,
                           prediction=h_pc_prediction,
                           n_comp=nho,
                           annotation=annotation,
                           labels=labels,
                           fig_file=fig_file)
            # Explained variance plots
            if exvar:
                i += 1
                annotation = [my_alphabet(i)]

                fig_file = os.path.join(hbase, 'roots_whpa', f'{root}_pca_exvar.png')
                explained_variance(h_pco.operator,
                                   n_comp=h_pco.n_pc_cut,
                                   thr=.85,
                                   annotation=annotation,
                                   fig_file=fig_file)

    @staticmethod
    def cca_vision(root: str = None,
                   folders=None):
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
            f_names = list(map(lambda fn: os.path.join(res_dir, f'{fn}.pkl'), ['cca', 'd_pca']))
            cca_operator, d_pco = list(map(joblib.load, f_names))
            h_pco = joblib.load(os.path.join(base_dir, 'h_pca.pkl'))

            h_pred = np.load(os.path.join(base_dir, 'roots_whpa', f'{root}.npy'))

            # Inspect transformation between physical and PC space
            dnc0 = d_pco.n_pc_cut
            hnc0 = h_pco.n_pc_cut

            # Cut desired number of PC components
            d_pc_training, d_pc_prediction = d_pco.pca_refresh(dnc0)
            h_pco.pca_test_fit_transform(h_pred, test_root=[root])
            h_pc_training, h_pc_prediction = h_pco.pca_refresh(hnc0)

            # CCA plots
            d_cca_training, h_cca_training = cca_operator.transform(d_pc_training, h_pc_training)
            d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T

            # Test 2/12 : Plot gaussian h cca
            # processing = TargetIO()
            # h_cca_training = processing.gaussian_distribution(h_cca_training)

            cca_plot(cca_operator,
                     d_cca_training,
                     h_cca_training,
                     d_pc_prediction,
                     h_pc_prediction,
                     sdir=os.path.join(os.path.dirname(res_dir), 'cca'))

            # CCA coefficient plot
            cca_coefficient = np.corrcoef(d_cca_training, h_cca_training, ).diagonal(offset=cca_operator.n_components)
            plt.plot(cca_coefficient, 'lightblue', zorder=1)
            plt.scatter(x=np.arange(len(cca_coefficient)),
                        y=cca_coefficient,
                        c=cca_coefficient,
                        alpha=1,
                        s=50,
                        cmap='coolwarm',
                        zorder=2)
            cb = plt.colorbar()
            cb.ax.set_title('R')
            plt.grid(alpha=.4, linewidth=.5, zorder=0)
            plt.xticks(np.arange(len(cca_coefficient)), np.arange(1, len(cca_coefficient) + 1))
            plt.tick_params(labelsize=5)
            plt.yticks([])
            # plt.title('Decrease of CCA correlation coefficient with component number')
            plt.ylabel('Correlation coefficient')
            plt.xlabel('Component number')
            plt.savefig(os.path.join(os.path.dirname(res_dir), 'cca', 'coefs.png'),
                        bbox_inches='tight',
                        dpi=300,
                        transparent=True)
            plt.close()

    @staticmethod
    def plot_whpa(root: str = None):
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
    def plot_pc_ba(root: str = None,
                   data: bool = False,
                   target: bool = False):
        """
        Comparison between original variables and the same variables back-transformed with n PCA components.
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
            hnc0 = h_pco.n_pc_cut
            # mplot.h_pca_inverse_plot(h_pco, hnc0, training=True, fig_dir=os.path.join(base_dir, 'control'))

            h_pred = np.load(os.path.join(base_dir, 'roots_whpa', f'{root}.npy'))  # Signed Distance
            # Cut desired number of PC components
            h_pco.pca_test_fit_transform(h_pred, test_root=[root])
            h_pco.pca_refresh(hnc0)
            mplot.h_pca_inverse_plot(h_pco,
                                     hnc0,
                                     training=False,
                                     fig_dir=jp(base_dir, 'roots_whpa'))

        # d
        if data:
            subdir = os.path.join(MySetup.Directories.forecasts_dir, root)
            listme = os.listdir(subdir)
            folders = list(filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))

            for f in folders:
                res_dir = os.path.join(subdir, f, 'obj')
                # Load objects
                d_pco = joblib.load(os.path.join(res_dir, 'd_pca.pkl'))
                dnc0 = d_pco.n_pc_cut  # Number of PC components
                d_pco.pca_refresh(dnc0)  # refresh based on dnc0
                setattr(d_pco, 'test_root', [root])
                # mplot.d_pca_inverse_plot(d_pco, dnc0, training=True,
                #                          fig_dir=os.path.join(os.path.dirname(res_dir), 'pca'))
                # Plot parameters for predictor
                xlabel = 'Observation index number'
                ylabel = 'Concentration ($g/m^{3})$'
                factor = 1000
                labelsize = 11
                mplot.d_pca_inverse_plot(d_pco,
                                         xlabel=xlabel,
                                         ylabel=ylabel,
                                         labelsize=labelsize,
                                         factor=factor,
                                         training=False,
                                         fig_dir=os.path.join(os.path.dirname(res_dir), 'pca'))
