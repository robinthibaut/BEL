#  Copyright (c) 2021. Robin Thibaut, Ghent University
import itertools
import os
from os.path import join as jp
import string

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from mpl_toolkits.axes_grid1 import make_axes_locatable

from experiment.spatial.distance import grid_parameters
from experiment.spatial.grid import contours_vertices, refine_machine
from experiment._core import MySetup
from experiment.spatial.grid import binary_stack
from experiment.toolbox import filesio
from sklearn.preprocessing import PowerTransformer

ftype = 'png'


def my_alphabet(az):
    """
    Method used to make custom figure annotations.
    :param az:
    :return:
    """
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
                 fig_file: str = None,
                 extra: list = None):
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
    :param extra: List of extra elements to be added on the final figure
    :return:
    """

    # Default parameters
    if colors is None:
        colors = ['w']
    if labels is None:
        labels = []
    if pec is None:
        pec = [None for _ in range(len(colors))]
    if extra is None:
        extra = []

    # Proxy figures (empty plots)
    proxys = [plt.plot([], marker, color=c, markeredgecolor=pec[i]) for i, c in enumerate(colors)]
    plt.legend([p[0] for p in proxys], labels, loc=loc, fontsize=fz)

    if legend1:
        plt.gca().add_artist(legend1)

    for el in extra:
        plt.gca().add_artist(el)

    if fig_file:
        filesio.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()


def proxy_annotate(annotation: list,
                   loc: int = 1,
                   fz: float = 11):
    """
    Places annotation (or title) within the figure box
    :param annotation: Must be a list of labels even of it only contains one label. Savvy ?
    :param fz: Fontsize
    :param loc: Location (default: 1 = upper right corner, 2 = upper left corner)
    :return:
    """

    legend_a = plt.legend(plt.plot([], linestyle=None, color='w', alpha=0, markeredgecolor=None),
                          annotation,
                          handlelength=0,
                          handletextpad=0,
                          fancybox=True,
                          loc=loc,
                          fontsize=fz)

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
    :param annotation: List of annotation(s)
    :param fig_file:
    :param show:
    :return:
    """
    plt.grid(alpha=0.1)
    if not n_comp:
        n_comp = pca.n_components_

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
    # Axes labels
    plt.xlabel('PC number', fontsize=11)
    plt.ylabel('Cumulative explained variance (%)', fontsize=11)
    # Legend
    legend_a = proxy_annotate(annotation=annotation, loc=2, fz=14)
    plt.gca().add_artist(legend_a)

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
    :param annotation: List of annotation(s)
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
    kde = False
    if kde:  # Option to plot KDE
        # Choose seaborn cmap
        # cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=.69, reverse=True)
        # cmap = sns.cubehelix_palette(start=6, rot=0, dark=0, light=.69, reverse=True, as_cmap=True)
        cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=.15, reverse=True)
        # KDE plot
        sns.kdeplot(np.arange(1, ut + 1), training.T[:ut][:, 1], cmap=cmap, n_levels=60, shade=True, vertical=True)
        plt.xlim(0.5, ut)  # Define x limits [start, end]
    # For each sample used for prediction:
    for sample_n in range(len(prediction)):
        # Select observation
        pc_obs = prediction[sample_n]
        # Create beautiful spline to follow prediction scores
        xnew = np.linspace(1, ut, 200)  # New points for plotting curve
        spl = make_interp_spline(np.arange(1, ut + 1), pc_obs.T[:ut], k=3)  # type: BSpline
        power_smooth = spl(xnew)
        # I forgot why I had to put '-1'
        plt.plot(xnew - 1,
                 power_smooth,
                 'red',
                 linewidth=1.2,
                 alpha=.9)

        plt.plot(pc_obs.T[:ut],  # Plot observations scores
                 'ro',
                 markersize=3,
                 markeredgecolor='k',
                 markeredgewidth=.4,
                 alpha=.8,
                 label=str(sample_n))

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


def cca_plot(cca_operator,
             d: np.array,
             h: np.array,
             d_pc_prediction: np.array,
             h_pc_prediction: np.array,
             sdir: str = None,
             show: bool = False):
    """
    CCA plots.
    Receives d, h PC components to be predicted, transforms them in CCA space and adds it to the plots.
    :param cca_operator: CCA operator (pickle)
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

            # %%  Watch out for the transpose operator.
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
            # Plot predicted canonical variate mean, or not
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
            plt.savefig(jp(sdir, 'cca{}.pdf'.format(i)), bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
        if show:
            plt.show()
            plt.close()


def whpa_plot(grf: float = None,
              well_comb: list = None,
              whpa: np.array = None,
              alpha: float = 0.4,
              halpha: float = None,
              lw: float = .5,
              bkg_field_array: np.array = None,
              vmin: float = None,
              vmax: float = None,
              x_lim: list = None,
              y_lim: list = None,
              xlabel: str = None,
              ylabel: str = None,
              cb_title: str = None,
              labelsize: float = 5,
              cmap: str = 'coolwarm',
              color: str = 'white',
              show_wells: bool = False,
              well_ids: list = None,
              title: str = None,
              annotation: list = None,
              fig_file: str = None,
              highlight: bool = False,
              show: bool = False):
    """
    Produces the WHPA plot, i.e. the zero-contour of the signed distance array.

    :param highlight: Boolean to display lines on top of filling between contours or not.
    :param annotation: List of annotations (str)
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
    :param whpa: np.array: Array containing grids of values whose 0 contour will be computed and plotted
    :param alpha: float: opacity of the 0 contour lines
    :param halpha: Alpha value for line plots if highlight is True
    :param lw: float: Line width
    :param color: str: Line color
    :param fig_file: str:
    :param show: bool:
    :param x_lim: [x_min, x_max] For the figure
    :param y_lim: [y_min, y_max] For the figure
    """

    # Get basic settings
    focus = MySetup.Focus()
    wells = MySetup.Wells()

    if well_comb is not None:
        wells.combination = well_comb

    if y_lim is None:
        ylim = focus.y_range
    else:
        ylim = y_lim
    if x_lim is None:
        xlim = focus.x_range
    else:
        xlim = x_lim
    if grf is None:
        grf = focus.cell_dim
    else:
        grf = grf

    nrow, ncol, x, y = refine_machine(focus.x_range, focus.y_range, grf)

    # Plot background
    if bkg_field_array is not None:
        plt.imshow(bkg_field_array,
                   extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                   vmin=vmin,
                   vmax=vmax,
                   cmap=cmap)
        cb = plt.colorbar()
        cb.ax.set_title(cb_title)

    if halpha is None:
        halpha = alpha

    # Plot results
    if whpa is None:
        whpa = []

    if len(whpa) > 1:  # New approach is to plot filled contours
        new_grf = 1  # Refine grid
        _, _, new_x, new_y = refine_machine(xlim,
                                            ylim,
                                            new_grf=new_grf)
        xys, nrow, ncol = grid_parameters(x_lim=focus.x_range,
                                          y_lim=focus.y_range,
                                          grf=new_grf)
        vertices = contours_vertices(x=x,
                                     y=y,
                                     arrays=whpa)
        b_low = binary_stack(xys=xys,
                             nrow=nrow,
                             ncol=ncol,
                             vertices=vertices)
        contour = plt.contourf(new_x,
                               new_y,
                               1 - b_low,  # Trick to be able to fill contours
                               [np.finfo(float).eps, 1 - np.finfo(float).eps],  # Use machine epsilon
                               colors=color,
                               alpha=alpha)
        if highlight:  # Also display curves
            for z in whpa:
                contour = plt.contour(x,
                                      y,
                                      z,
                                      [0],
                                      colors=color,
                                      linewidths=lw,
                                      alpha=halpha)

    else:  # If only one WHPA to display
        contour = plt.contour(x,
                              y,
                              whpa[0],
                              [0],
                              colors=color,
                              linewidths=lw,
                              alpha=halpha)

    # Grid
    plt.grid(color='c',
             linestyle='-',
             linewidth=.5,
             alpha=.2)

    # Plot wells
    well_legend = None
    if show_wells:
        plot_wells(wells,
                   well_ids=well_ids,
                   markersize=7)
        well_legend = plt.legend(fontsize=11)

    # Plot limits
    if x_lim is None:
        plt.xlim(xlim[0], xlim[1])
    else:
        plt.xlim(x_lim[0], x_lim[1])
    if y_lim is None:
        plt.ylim(ylim[0], ylim[1])
    else:
        plt.ylim(y_lim[0], y_lim[1])

    if title:
        plt.title(title)

    plt.xlabel(xlabel, fontsize=labelsize)
    plt.ylabel(ylabel, fontsize=labelsize)

    # Tick size
    plt.tick_params(labelsize=labelsize, colors='k')

    if annotation:
        legend = proxy_annotate(annotation=annotation, fz=14, loc=2)
        plt.gca().add_artist(legend)

    if fig_file:
        filesio.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
    if show:
        plt.show()
        plt.close()

    return contour, well_legend


def post_examination(root: str,
                     xlim: list = None,
                     ylim: list = None,
                     show: bool = False):
    focus = MySetup.Focus()
    if xlim is None:
        xlim = focus.x_range
    if ylim is None:
        ylim = focus.y_range  # [335, 700]
    md = MySetup.Directories()
    ndir = jp(md.forecasts_dir, 'base', 'roots_whpa', f'{root}.npy')
    sdir = os.path.dirname(ndir)
    nn = np.load(ndir)
    whpa_plot(whpa=nn,
              x_lim=xlim,
              y_lim=ylim,
              labelsize=11,
              alpha=1,
              xlabel='X(m)',
              ylabel='Y(m)',
              cb_title='SD(m)',
              annotation=['B'],
              bkg_field_array=np.flipud(nn[0]),
              color='black',
              cmap='coolwarm')

    # legend = proxy_annotate(annotation=['B'], loc=2, fz=14)
    # plt.gca().add_artist(legend)

    plt.savefig(jp(sdir, f'{root}_SD.pdf'),
                dpi=300,
                bbox_inches='tight',
                transparent=True)
    if show:
        plt.show()
    plt.close()


def h_pca_inverse_plot(pca_o,
                       training: bool = True,
                       fig_dir: str = None,
                       show: bool = False):
    """
    Plot used to compare the reproduction of the original physical space after PCA transformation
    :param pca_o: signed distance PCA operator
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

        whpa_plot(whpa=h_to_plot,
                  color='red',
                  alpha=1,
                  lw=2)

        v_pred = pca_o.custom_inverse_transform(v_pc)

        whpa_plot(whpa=v_pred.reshape(1, shape[1], shape[2]),
                  color='blue',
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
            plt.savefig(jp(fig_dir, f'{r}_h.pdf'), dpi=300, transparent=True)
            plt.close()

        if show:
            plt.show()
            plt.close()


def plot_results(d: bool = True,
                 h: bool = True,
                 root: str = None,
                 folder: str = None,
                 annotation: list = None):
    """
    Plots forecasts results in the 'uq' folder
    :param annotation: List of annotations
    :param h: Boolean to plot target or not
    :param d: Boolean to plot predictor or not
    :param root: str: Forward ID
    :param folder: str: Well combination. '123456', '1'...
    :return:
    """
    # Directory
    md = jp(MySetup.Directories.forecasts_dir, root, folder)

    # Wells
    wells = MySetup.Wells()
    wells_id = list(wells.wells_data.keys())
    cols = [wells.wells_data[w]['color'] for w in wells_id if 'pumping' not in w]

    # CCA pickle
    cca_operator = joblib.load(jp(md, 'obj', 'cca.pkl'))

    # d pca pickle
    d_pco = joblib.load(jp(md, 'obj', 'd_pca.pkl'))

    # h PCA pickle
    hbase = jp(MySetup.Directories.forecasts_dir, 'base')
    pcaf = jp(hbase, 'h_pca.pkl')
    h_pco = joblib.load(pcaf)

    if d:
        # Curves - d
        # Plot curves
        sdir = jp(md, 'data')

        tc = d_pco.training_physical.reshape(d_pco.training_shape)
        tcp = d_pco.predict_physical.reshape(d_pco.obs_shape)
        tc = np.concatenate((tc, tcp), axis=0)

        # Plot parameters for predictor
        xlabel = 'Observation index number'
        ylabel = 'Concentration ($g/m^{3})$'
        factor = 1000
        labelsize = 11

        curves(cols, tc=tc,
               sdir=sdir,
               xlabel=xlabel,
               ylabel=ylabel,
               factor=factor,
               labelsize=labelsize,
               highlight=[len(tc) - 1])

        curves(cols, tc=tc,
               sdir=sdir,
               xlabel=xlabel,
               ylabel=ylabel,
               factor=factor,
               labelsize=labelsize,
               highlight=[len(tc) - 1],
               ghost=True,
               title='curves_ghost')

        curves_i(cols, tc=tc,
                 xlabel=xlabel,
                 ylabel=ylabel,
                 factor=factor,
                 labelsize=labelsize,
                 sdir=sdir,
                 highlight=[len(tc) - 1])

    if h:
        # WHP - h test + training
        fig_dir = jp(hbase, 'roots_whpa')
        ff = jp(fig_dir, f'{root}.pdf')  # figure name
        h_test = np.load(jp(fig_dir, f'{root}.npy')).reshape(h_pco.obs_shape)
        h_training = h_pco.training_physical.reshape(h_pco.training_shape)
        # Plots target training + prediction
        whpa_plot(whpa=h_training, color='blue', alpha=.5)
        whpa_plot(whpa=h_test, color='r', lw=2, alpha=.8, xlabel='X(m)', ylabel='Y(m)', labelsize=11)
        colors = ['blue', 'red']
        labels = ['Training', 'Test']
        legend = proxy_annotate(annotation=['C'], loc=2, fz=14)
        proxy_legend(legend1=legend, colors=colors, labels=labels, fig_file=ff)

        # WHPs
        ff = jp(md,
                'uq',
                f'{root}_cca_{cca_operator.n_components}.pdf')
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
        _, well_legend = whpa_plot(whpa=h_training,
                                   alpha=.5,
                                   lw=.5,
                                   color=colors[0],
                                   show_wells=True,
                                   well_ids=well_ids,
                                   show=False)

        # Samples
        whpa_plot(whpa=forecast_posterior,
                  color=colors[1],
                  lw=1,
                  alpha=.8,
                  highlight=True,
                  show=False)

        # True test
        whpa_plot(whpa=h_test,
                  color=colors[2],
                  lw=.8,
                  alpha=1,
                  x_lim=[800, 1200],
                  xlabel='X(m)',
                  ylabel='Y(m)',
                  labelsize=11)

        # Other tricky operation to add annotation
        legend_an = proxy_annotate(annotation=annotation,
                                   loc=2,
                                   fz=14)

        # Tricky operation to add a second legend:
        proxy_legend(legend1=well_legend,
                     extra=[legend_an],
                     colors=colors,
                     labels=labels,
                     fig_file=ff)


def plot_K_field(root: str = None,
                 wells=None,
                 deprecated: bool = True):
    if wells is None:
        wells = MySetup.Wells()

    matrix = np.load(jp(MySetup.Directories.hydro_res_dir, root, 'hk0.npy'))
    grid_dim = MySetup.GridDimensions
    extent = (grid_dim.xo, grid_dim.x_lim, grid_dim.yo, grid_dim.y_lim)

    hkf = jp(MySetup.Directories.forecasts_dir, root, 'k_field.png')

    if deprecated:
        # HK field
        plt.figure()
        ax = plt.gca()
        im = ax.imshow(np.log10(matrix), cmap='coolwarm', extent=extent)
        plt.xlabel('X(m)', fontsize=11)
        plt.ylabel('Y(m)', fontsize=11)
        plot_wells(wells, markersize=3.5)
        # well_legend = plt.legend(fontsize=11, loc=2, framealpha=.6)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.ax.set_title('$Log_{10} m/s$')
        plt.savefig(hkf,
                    bbox_inches='tight',
                    dpi=300,
                    transparent=True)
        plt.close()


def mode_histo(colors: list,
               an_i: int,
               wm: np.array,
               fig_name: str = 'average'):
    alphabet = string.ascii_uppercase
    wid = list(map(str, MySetup.Wells.combination))  # Wel identifiers (n)

    modes = []  # Get MHD corresponding to each well's mode
    for i, m in enumerate(wm):  # For each well, look up its MHD distribution
        count, values = np.histogram(m, bins='fd')
        # (Freedman Diaconis Estimator)
        # Robust (resilient to outliers) estimator that takes into account data variability and data size.
        # https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
        idm = np.argmax(count)
        mode = values[idm]
        modes.append(mode)

    modes = np.array(modes)  # Scale modes
    modes -= np.mean(modes)

    # Bar plot
    plt.bar(np.arange(1, 7), -modes, color=colors)
    plt.title('Amount of information of each well')
    plt.xlabel('Well ID')
    plt.ylabel('Opposite deviation from mode\'s mean')
    plt.grid(color='#95a5a6', linestyle='-', linewidth=.5, axis='y', alpha=0.7)

    legend_a = proxy_annotate(annotation=[alphabet[an_i + 1]], loc=2, fz=14)
    plt.gca().add_artist(legend_a)

    plt.savefig(os.path.join(MySetup.Directories.forecasts_dir, f'{fig_name}_well_mode.pdf'), dpi=300,
                transparent=True)
    plt.close()
    # plt.show()

    # Plot histogram
    for i, m in enumerate(wm):
        sns.kdeplot(m, color=f'{colors[i]}', shade=True, linewidth=2)
    plt.title('Summed MHD distribution for each well')
    plt.xlabel('Summed MHD')
    plt.ylabel('KDE')
    legend_1 = plt.legend(wid, loc=1)
    plt.gca().add_artist(legend_1)
    plt.grid(alpha=0.2)

    legend_a = proxy_annotate(annotation=[alphabet[an_i]], loc=2, fz=14)
    plt.gca().add_artist(legend_a)

    plt.savefig(os.path.join(MySetup.Directories.forecasts_dir, f'{fig_name}_hist.pdf'), dpi=300, transparent=True)
    plt.close()
    # plt.show()

    # %% Facet histograms
    # ids = np.array(np.concatenate([np.ones(wm.shape[1]) * i for i in range(1, 7)]), dtype='int')
    # master = wm.flatten()
    #
    # data = np.concatenate([[master], [ids]], axis=0)
    #
    # master_x = pd.DataFrame(data=data.T, columns=['MHD', 'well'])
    # master_x['well'] = np.array(ids)
    # g = sns.FacetGrid(master_x,  # the dataframe to pull from
    #                   row="well",
    #                   hue="well",
    #                   aspect=3,  # aspect * height = width
    #                   height=1.5,  # height of each subplot
    #                   palette=colors  # google colors
    #                   )
    #
    # g.map(sns.kdeplot, "MHD", shade=True, alpha=1, lw=1.5)
    # g.map(plt.axhline, y=0, lw=4)
    # for ax in g.axes:
    #     ax[0].set_xlim((500, 1000))
    #
    # def label(x, color, label):
    #     ax = plt.gca()  # get the axes of the current object
    #     ax.text(0, .2,  # location of text
    #             label,  # text label
    #             fontweight="bold", color=color, size=20,  # text attributes
    #             ha="left", va="center",  # alignment specifications
    #             transform=ax.transAxes)  # specify axes of transformation)
    #
    # g.map(label, "MHD")  # the function counts as a plotting object!
    #
    # sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # g.fig.subplots_adjust(hspace=-.25)
    #
    # g.set_titles("")  # set title to blank
    # g.set_xlabels(color="white")
    # g.set_xticklabels(color='white', fontsize=14)
    # g.set(yticks=[])  # set y ticks to blank
    # g.despine(bottom=True, left=True)  # remove 'spines'
    #
    # plt.savefig(os.path.join(MySetup.Directories.forecasts_dir, f'{fig_name}_facet.pdf'), dpi=300, transparent=True)
    # plt.close()
    # plt.show()


def curves(cols: list,
           tc: np.array,
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
                plt.plot(tc[i][t] * factor, color=cols[t], linewidth=2, alpha=1)
            elif not ghost:
                plt.plot(tc[i][t] * factor, color=cols[t], linewidth=.2, alpha=0.5)

    plt.grid(linewidth=.3, alpha=.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tick_params(labelsize=labelsize)
    if sdir:
        filesio.dirmaker(sdir)
        plt.savefig(jp(sdir, f'{title}.pdf'), dpi=300, transparent=True)
        plt.close()
    if show:
        plt.show()
        plt.close()


def curves_i(cols, tc,
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
                plt.plot(tc[i][t] * factor, color=cols[t], linewidth=.2, alpha=0.5)
        colors = [cols[t], 'k']
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
            plt.savefig(jp(sdir, f'{title}_{t + 1}.pdf'), dpi=300, transparent=True)
            plt.close()
        if show:
            plt.show()
            plt.close()


def plot_wells(wells,
               well_ids=None,
               markersize: float = 4.):
    if well_ids is None:
        comb = [0] + list(wells.combination)
    else:
        comb = well_ids
    # comb = [0] + list(self.wells.combination)
    # comb = [0] + list(self.wells.combination)
    keys = [list(wells.wells_data.keys())[i] for i in comb]
    wbd = {k: wells.wells_data[k] for k in keys if k in wells.wells_data}
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


def plot_head_field(root: str = None):
    matrix = np.load(jp(MySetup.Directories.hydro_res_dir, root, 'whpa_heads.npy'))
    grid_dim = MySetup.GridDimensions
    extent = (grid_dim.xo, grid_dim.x_lim, grid_dim.yo, grid_dim.y_lim)

    hkf = jp(MySetup.Directories.forecasts_dir, root, 'heads_field.png')

    # HK field
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(matrix[0], cmap='Blues_r', extent=extent)
    plt.xlabel('X(m)', fontsize=11)
    plt.ylabel('Y(m)', fontsize=11)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.set_title('Head (m)')

    plt.savefig(hkf,
                bbox_inches='tight',
                dpi=300,
                transparent=True)
    plt.close()


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
        h_pca_inverse_plot(pca_o=h_pco,
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
            d_pca_inverse_plot(d_pco,
                               xlabel=xlabel,
                               ylabel=ylabel,
                               labelsize=labelsize,
                               factor=factor,
                               training=False,
                               fig_dir=os.path.join(os.path.dirname(res_dir), 'pca'))


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
    fobj = os.path.join(MySetup.Directories.forecasts_dir, 'base', 'h_pca.pkl')
    h = joblib.load(fobj)
    h_training = h.training_physical.reshape(h.training_shape)

    whpa_plot(whpa=h_training,
              highlight=True,
              halpha=.5,
              lw=.1,
              color='darkblue',
              alpha=.5)

    if root is not None:
        h_pred = np.load(os.path.join(base_dir, 'roots_whpa', f'{root}.npy'))
        whpa_plot(whpa=h_pred,
                  color='darkred',
                  lw=1,
                  alpha=1,
                  annotation=['C'],
                  xlabel='X(m)',
                  ylabel='Y(m)',
                  labelsize=11)

        labels = ['Training', 'Test']
        legend = proxy_annotate(annotation=['C'], loc=2, fz=14)
        proxy_legend(legend1=legend,
                     colors=['darkblue', 'darkred'],
                     labels=labels,
                     fig_file=os.path.join(MySetup.Directories.forecasts_dir, root, 'whpa_training.pdf'))


def cca_vision(root: str = None,
               folders: list = None):
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

        # Add annotation
        legend = proxy_annotate(annotation=['D'], fz=14, loc=1)
        plt.gca().add_artist(legend)

        plt.savefig(os.path.join(os.path.dirname(res_dir), 'cca', 'coefs.pdf'),
                    bbox_inches='tight',
                    dpi=300,
                    transparent=True)
        plt.close()


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

    if d:
        for f in folders:
            dfig = os.path.join(subdir, f, 'pca')
            # For d only
            pcaf = os.path.join(subdir, f, 'obj', 'd_pca.pkl')
            d_pco = joblib.load(pcaf)
            fig_file = os.path.join(dfig, 'd_scores.pdf')
            if scores:
                pca_scores(training=d_pco.training_pc,
                           prediction=d_pco.predict_pc,
                           n_comp=d_pco.n_pc_cut,
                           annotation=['E'],
                           labels=labels,
                           fig_file=fig_file)
            # Explained variance plots
            if exvar:
                fig_file = os.path.join(dfig, 'd_exvar.pdf')
                explained_variance(d_pco.operator,
                                   n_comp=d_pco.n_pc_cut,
                                   thr=.8,
                                   annotation=['C'],
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
        fig_file = os.path.join(hbase, 'roots_whpa', f'{root}_pca_scores.pdf')
        if scores:
            pca_scores(training=h_pc_training,
                       prediction=h_pc_prediction,
                       n_comp=nho,
                       annotation=['F'],
                       labels=labels,
                       fig_file=fig_file)
        # Explained variance plots
        if exvar:
            fig_file = os.path.join(hbase, 'roots_whpa', f'{root}_pca_exvar.pdf')
            explained_variance(h_pco.operator,
                               n_comp=h_pco.n_pc_cut,
                               thr=.85,
                               annotation=['D'],
                               fig_file=fig_file)


def check_root(xlim: list,
               ylim: list,
               root: list):
    """
    Plots raw data of folder 'root'
    :param xlim:
    :param ylim:
    :param root:
    :return:
    """
    bkt, whpa, _ = filesio.data_loader(roots=root)
    whpa = whpa.squeeze()
    # self.curves_i(bkt, show=True)  # This function will not work

    plt.plot(whpa[:, 0], whpa[:, 1], 'wo')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


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
                     marker='-',
                     loc=1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tick_params(labelsize=labelsize)

        # Increase y axis by a small percentage for annotation in upper left corner
        yrange = np.max(to_plot * factor) * 1.15
        plt.ylim([0, yrange])

        if fig_dir is not None:
            filesio.dirmaker(fig_dir)
            plt.savefig(jp(fig_dir, f'{r}_d.pdf'), dpi=300, transparent=True)
            plt.close()
        if show:
            plt.show()
            plt.close()


def hydro_examination(root: str):
    md = MySetup.Directories()
    ep = jp(md.hydro_res_dir, root, 'tracking_ep.npy')
    epxy = np.load(ep)

    plt.plot(epxy[:, 0], epxy[:, 1], 'ko')
    # seed = np.random.randint(2**32 - 1)
    # np.random.seed(seed)
    # sample = np.random.randint(144, size=10)
    sample = np.array([94, 10, 101, 29, 43, 116, 100, 40, 72])
    for i in sample:
        plt.text(epxy[i, 0] + 4,
                 epxy[i, 1] + 4,
                 i,
                 color='black',
                 fontsize=11,
                 weight='bold',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=.5', alpha=.7))
        # plt.annotate(i, (epxy[i, 0] + 4, epxy[i, 1] + 4), fontsize=14, weight='bold', color='r')

    plt.grid(alpha=.5)
    plt.xlim([870, 1080])
    plt.ylim([415, 600])
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    plt.tick_params(labelsize=11)

    legend = proxy_annotate(annotation=['A'],
                            loc=2,
                            fz=14)
    plt.gca().add_artist(legend)

    plt.savefig(jp(md.forecasts_dir, 'base', 'roots_whpa', f'{root}_ep.pdf'),
                dpi=300,
                bbox_inches='tight',
                transparent=True)
    plt.show()


def despine(fig=None, ax=None, top=True, right=True, left=False,
            bottom=False, offset=None, trim=False):
    """Remove the top and right spines from plot(s).

    fig : matplotlib figure, optional
        Figure to despine all axes of, defaults to the current figure.
    ax : matplotlib axes, optional
        Specific axes object to despine. Ignored if fig is provided.
    top, right, left, bottom : boolean, optional
        If True, remove that spine.
    offset : int or dict, optional
        Absolute distance, in points, spines should be moved away
        from the axes (negative values move spines inward). A single value
        applies to all spines; a dict can be used to set offset values per
        side.
    trim : bool, optional
        If True, limit spines to the smallest and largest major tick
        on each non-despined axis.

    Returns
    -------
    None

    """
    # Get references to the axes we want
    if fig is None and ax is None:
        axes = plt.gcf().axes
    elif fig is not None:
        axes = fig.axes
    elif ax is not None:
        axes = [ax]

    for ax_i in axes:
        for side in ["top", "right", "left", "bottom"]:
            # Toggle the spine objects
            is_visible = not locals()[side]
            ax_i.spines[side].set_visible(is_visible)
            if offset is not None and is_visible:
                try:
                    val = offset.get(side, 0)
                except AttributeError:
                    val = offset
                ax_i.spines[side].set_position(('outward', val))

        # Potentially move the ticks
        if left and not right:
            maj_on = any(
                t.tick1line.get_visible()
                for t in ax_i.yaxis.majorTicks
            )
            min_on = any(
                t.tick1line.get_visible()
                for t in ax_i.yaxis.minorTicks
            )
            ax_i.yaxis.set_ticks_position("right")
            for t in ax_i.yaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.yaxis.minorTicks:
                t.tick2line.set_visible(min_on)

        if bottom and not top:
            maj_on = any(
                t.tick1line.get_visible()
                for t in ax_i.xaxis.majorTicks
            )
            min_on = any(
                t.tick1line.get_visible()
                for t in ax_i.xaxis.minorTicks
            )
            ax_i.xaxis.set_ticks_position("top")
            for t in ax_i.xaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.xaxis.minorTicks:
                t.tick2line.set_visible(min_on)

        if trim:
            # clip off the parts of the spines that extend past major ticks
            xticks = np.asarray(ax_i.get_xticks())
            if xticks.size:
                firsttick = np.compress(xticks >= min(ax_i.get_xlim()),
                                        xticks)[0]
                lasttick = np.compress(xticks <= max(ax_i.get_xlim()),
                                       xticks)[-1]
                ax_i.spines['bottom'].set_bounds(firsttick, lasttick)
                ax_i.spines['top'].set_bounds(firsttick, lasttick)
                newticks = xticks.compress(xticks <= lasttick)
                newticks = newticks.compress(newticks >= firsttick)
                ax_i.set_xticks(newticks)

            yticks = np.asarray(ax_i.get_yticks())
            if yticks.size:
                firsttick = np.compress(yticks >= min(ax_i.get_ylim()),
                                        yticks)[0]
                lasttick = np.compress(yticks <= max(ax_i.get_ylim()),
                                       yticks)[-1]
                ax_i.spines['left'].set_bounds(firsttick, lasttick)
                ax_i.spines['right'].set_bounds(firsttick, lasttick)
                newticks = yticks.compress(yticks <= lasttick)
                newticks = newticks.compress(newticks >= firsttick)
                ax_i.set_yticks(newticks)