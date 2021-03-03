#  Copyright (c) 2021. Robin Thibaut, Ghent University
import itertools
import math
import operator
import os
import string
from functools import reduce
from os.path import join as jp

import flopy
import joblib
import numpy as np
import seaborn as sns
import vtk
from flopy.export import vtk as vtk_flow
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ma
from scipy.interpolate import BSpline, make_interp_spline

import experiment.algorithms.spatial
import experiment.algorithms.statistics
import experiment.utils
from experiment.algorithms import statistics as stats
from experiment.algorithms.spatial import (binary_stack, contours_vertices,
                                           grid_parameters, refine_machine)
from experiment.core import Setup
from experiment.utils import reload_trained_model

ftype = "png"


def my_alphabet(az):
    """
    Method used to make custom figure annotations.
    :param az:
    :return:
    """
    alphabet = string.ascii_uppercase
    extended_alphabet = [
        "".join(i) for i in list(itertools.permutations(alphabet, 2))
    ]

    if az <= 25:
        sub = alphabet[az]
    else:
        j = az - 26
        sub = extended_alphabet[j]

    return sub


def proxy_legend(
    legend1=None,
    colors: list = None,
    labels: list = None,
    loc: int = 4,
    marker: list = None,
    pec: list = None,
    fz: float = 11,
    fig_file: str = None,
    extra: list = None,
    obj=None,
):
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

    if obj is None:
        obj = plt
    # Default parameters
    if colors is None:
        colors = ["w"]
    if labels is None:
        labels = []
    if pec is None:
        pec = [None for _ in range(len(colors))]
    if extra is None:
        extra = []
    if marker is None:
        marker = ["-" for _ in range(len(colors))]

    # Proxy figures (empty plots)
    proxys = [
        plt.plot([], marker[i], color=c, markeredgecolor=pec[i])
        for i, c in enumerate(colors)
    ]

    obj.legend([p[0] for p in proxys], labels, loc=loc, fontsize=fz)

    if legend1:
        try:
            obj.gca().add_artist(legend1)
        except AttributeError:
            obj.add_artist(legend1)

    for el in extra:
        try:
            obj.gca().add_artist(el)
        except AttributeError:
            obj.add_artist(el)

    if fig_file:
        experiment.utils.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, bbox_inches="tight", dpi=300, transparent=True)
        plt.close()


def proxy_annotate(annotation: list = None,
                   loc: int = 1,
                   fz: float = 11,
                   obj=None):
    """
    Places annotation (or title) within the figure box
    :param annotation: Must be a list of labels even of it only contains one label. Savvy ?
    :param fz: Fontsize
    :param loc: Location (default: 1 = upper right corner, 2 = upper left corner)
    :return:
    """
    if obj is None:
        obj = plt

    legend_a = obj.legend(
        plt.plot([], linestyle=None, color="w", alpha=0, markeredgecolor=None),
        annotation,
        handlelength=0,
        handletextpad=0,
        fancybox=True,
        loc=loc,
        fontsize=fz,
    )

    return legend_a


def explained_variance(
    pca,
    n_comp: int = 0,
    thr: float = 1.0,
    annotation: list = None,
    fig_file: str = None,
    show: bool = False,
):
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
    plt.bar(
        np.arange(n_comp),
        np.cumsum(pca.explained_variance_ratio_[:n_comp]) * 100,
        color="m",
        alpha=0.1,
    )
    # line for aesthetics
    plt.plot(
        np.arange(n_comp),
        np.cumsum(pca.explained_variance_ratio_[:n_comp]) * 100,
        "-o",
        linewidth=0.5,
        markersize=1.5,
        alpha=0.8,
    )
    # Axes labels
    plt.xlabel("PC number", fontsize=11)
    plt.ylabel("Cumulative explained variance (%)", fontsize=11)
    # Legend
    legend_a = proxy_annotate(annotation=annotation, loc=2, fz=14)
    plt.gca().add_artist(legend_a)

    if fig_file:
        experiment.utils.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, dpi=300, transparent=True)
        plt.close()
    if show:
        plt.show()
        plt.close()


def pca_scores(
    training,
    prediction,
    n_comp: int,
    annotation: list,
    fig_file: str = None,
    labels: bool = True,
    show: bool = False,
):
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
    # Ticks
    plt.xticks(np.arange(0, n_comp, 5), fontsize=11)
    # Plot all training scores
    plt.plot(training.T[:n_comp], "ob", markersize=3, alpha=0.05)
    # plt.plot(training.T[:ut], '+w', markersize=.5, alpha=0.2)
    kde = False
    if kde:  # Option to plot KDE
        # Choose seaborn cmap
        # cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=.69, reverse=True)
        # cmap = sns.cubehelix_palette(start=6, rot=0, dark=0, light=.69, reverse=True, as_cmap=True)
        cmap = sns.cubehelix_palette(as_cmap=True,
                                     dark=0,
                                     light=0.15,
                                     reverse=True)
        # KDE plot
        sns.kdeplot(
            np.arange(1, n_comp + 1),
            training.T[:n_comp][:, 1],
            cmap=cmap,
            n_levels=60,
            shade=True,
            vertical=True,
        )
        plt.xlim(0.5, n_comp)  # Define x limits [start, end]
    # For each sample used for prediction:
    for sample_n in range(len(prediction)):
        # Select observation
        pc_obs = prediction[sample_n]
        # Create beautiful spline to follow prediction scores
        xnew = np.linspace(1, n_comp, 200)  # New points for plotting curve
        spl = make_interp_spline(np.arange(1, n_comp + 1),
                                 pc_obs.T[:n_comp],
                                 k=3)  # type: BSpline
        power_smooth = spl(xnew)
        # I forgot why I had to put '-1'
        plt.plot(xnew - 1, power_smooth, "red", linewidth=1.2, alpha=0.9)

        plt.plot(
            pc_obs.T[:n_comp],  # Plot observations scores
            "ro",
            markersize=3,
            markeredgecolor="k",
            markeredgewidth=0.4,
            alpha=0.8,
            label=str(sample_n),
        )

    if labels:
        plt.title("Principal Components of training and test dataset")
        plt.xlabel("PC number")
        plt.ylabel("PC")
    plt.tick_params(labelsize=11)
    # Add legend
    # Add title inside the box
    legend_a = proxy_annotate(annotation=annotation, loc=2, fz=14)
    proxy_legend(
        legend1=legend_a,
        colors=["blue", "red"],
        labels=["Training", "Test"],
        marker=["o", "o"],
    )

    if fig_file:
        experiment.utils.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, dpi=300, transparent=True)
        plt.close()
    if show:
        plt.show()


def cca_plot(
    cca_operator,
    d: np.array,
    h: np.array,
    d_pc_prediction: np.array,
    h_pc_prediction: np.array,
    sdir: str = None,
    show: bool = False,
):
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

    cca_coefficient = np.corrcoef(d, h).diagonal(
        offset=cca_operator.n_components)  # Gets correlation coefficient

    # CCA plots for each observation:
    for i in range(cca_operator.n_components):
        comp_n = i
        for sample_n in range(len(d_pc_prediction)):  # For each 'observation'
            pass

        subtitle = my_alphabet(i)

        # Add title inside the box
        an = [f"{subtitle}. Pair {i + 1} - R = {round(cca_coefficient[i], 3)}"]
        legend_a = proxy_annotate(annotation=an, loc=2, fz=14)

        proxy_legend(
            legend1=legend_a,
            colors=["black", "white"],
            labels=["Training", "Test"],
            marker=["o", "o"],
            pec=["k", "k"],
        )

        if sdir:
            experiment.utils.dirmaker(sdir)
            plt.savefig(
                jp(sdir, "cca{}.pdf".format(i)),
                bbox_inches="tight",
                dpi=300,
                transparent=True,
            )
            plt.close()
        if show:
            plt.show()
            plt.close()


def whpa_plot(
    grf: float = None,
    well_comb: list = None,
    whpa: np.array = None,
    alpha: float = 0.4,
    halpha: float = None,
    lw: float = 0.5,
    bkg_field_array: np.array = None,
    vmin: float = None,
    vmax: float = None,
    x_lim: list = None,
    y_lim: list = None,
    xlabel: str = None,
    ylabel: str = None,
    cb_title: str = None,
    labelsize: float = 5,
    cmap: str = "coolwarm",
    color: str = "white",
    show_wells: bool = False,
    well_ids: list = None,
    title: str = None,
    annotation: list = None,
    fig_file: str = None,
    highlight: bool = False,
    show: bool = False,
):
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
    focus = Setup.Focus()
    wells = Setup.Wells()

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
        plt.imshow(
            bkg_field_array,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        cb = plt.colorbar()
        cb.ax.set_title(cb_title)

    if halpha is None:
        halpha = alpha

    # Plot results
    if whpa is None:
        whpa = []

    if len(whpa) > 1:  # New approach is to plot filled contours
        new_grf = 1  # Refine grid
        _, _, new_x, new_y = refine_machine(xlim, ylim, new_grf=new_grf)
        xys, nrow, ncol = grid_parameters(x_lim=focus.x_range,
                                          y_lim=focus.y_range,
                                          grf=new_grf)
        vertices = contours_vertices(x=x, y=y, arrays=whpa)
        b_low = binary_stack(xys=xys, nrow=nrow, ncol=ncol, vertices=vertices)
        contour = plt.contourf(
            new_x,
            new_y,
            1 - b_low,  # Trick to be able to fill contours
            # Use machine epsilon
            [np.finfo(float).eps, 1 - np.finfo(float).eps],
            colors=color,
            alpha=alpha,
        )
        if highlight:  # Also display curves
            for z in whpa:
                contour = plt.contour(x,
                                      y,
                                      z, [0],
                                      colors=color,
                                      linewidths=lw,
                                      alpha=halpha)

    else:  # If only one WHPA to display
        contour = plt.contour(x,
                              y,
                              whpa[0], [0],
                              colors=color,
                              linewidths=lw,
                              alpha=halpha)

    # Grid
    plt.grid(color="c", linestyle="-", linewidth=0.5, alpha=0.2)

    # Plot wells
    well_legend = None
    if show_wells:
        plot_wells(wells, well_ids=well_ids, markersize=7)
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
    plt.tick_params(labelsize=labelsize, colors="k")

    if annotation:
        legend = proxy_annotate(annotation=annotation, fz=14, loc=2)
        plt.gca().add_artist(legend)

    if fig_file:
        experiment.utils.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, bbox_inches="tight", dpi=300, transparent=True)
        plt.close()
    if show:
        plt.show()
        plt.close()

    return contour, well_legend


def post_examination(root: str,
                     xlim: list = None,
                     ylim: list = None,
                     show: bool = False):
    focus = Setup.Focus()
    if xlim is None:
        xlim = focus.x_range
    if ylim is None:
        ylim = focus.y_range  # [335, 700]
    md = Setup.Directories()
    ndir = jp(md.forecasts_dir, "base", "roots_whpa", f"{root}.npy")
    sdir = os.path.dirname(ndir)
    nn = np.load(ndir)
    whpa_plot(
        whpa=nn,
        x_lim=xlim,
        y_lim=ylim,
        labelsize=11,
        alpha=1,
        xlabel="X(m)",
        ylabel="Y(m)",
        cb_title="SD(m)",
        annotation=["B"],
        bkg_field_array=np.flipud(nn[0]),
        color="black",
        cmap="coolwarm",
    )

    # legend = proxy_annotate(annotation=['B'], loc=2, fz=14)
    # plt.gca().add_artist(legend)

    plt.savefig(jp(sdir, f"{root}_SD.pdf"),
                dpi=300,
                bbox_inches="tight",
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
            h_to_plot = np.copy(pca_o.training_physical[i].reshape(
                1, shape[1], shape[2]))
        else:
            h_to_plot = np.copy(pca_o.predict_physical[i].reshape(
                1, shape[1], shape[2]))

        whpa_plot(whpa=h_to_plot, color="red", alpha=1, lw=2)

        v_pred = pca_o.custom_inverse_transform(v_pc)

        whpa_plot(
            whpa=v_pred.reshape(1, shape[1], shape[2]),
            color="blue",
            alpha=1,
            lw=2,
            labelsize=11,
            xlabel="X(m)",
            ylabel="Y(m)",
            x_lim=[850, 1100],
            y_lim=[350, 650],
        )

        # Add title inside the box
        an = ["B"]

        legend_a = proxy_annotate(annotation=an, loc=2, fz=14)

        proxy_legend(
            legend1=legend_a,
            colors=["red", "blue"],
            labels=["Physical", "Back transformed"],
            marker=["-", "-"],
        )

        if fig_dir is not None:
            experiment.utils.dirmaker(fig_dir)
            plt.savefig(jp(fig_dir, f"{r}_h.pdf"), dpi=300, transparent=True)
            plt.close()

        if show:
            plt.show()
            plt.close()


def plot_results(
    d: bool = True,
    h: bool = True,
    root: str = None,
    folder: str = None,
    annotation: list = None,
):
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
    md = jp(Setup.Directories.forecasts_dir, root, folder)

    # Wells
    wells = Setup.Wells()
    wells_id = list(wells.wells_data.keys())
    cols = [
        wells.wells_data[w]["color"] for w in wells_id if "pumping" not in w
    ]

    # CCA pickle
    cca_operator = joblib.load(jp(md, "obj", "cca.pkl"))

    # d pca pickle
    d_pco = joblib.load(jp(md, "obj", "d_pca.pkl"))

    # h PCA pickle
    hbase = jp(Setup.Directories.forecasts_dir, "base")
    pcaf = jp(hbase, "h_pca.pkl")
    h_pco = joblib.load(pcaf)

    if d:
        # Curves - d
        # Plot curves
        sdir = jp(md, "data")

        tc = d_pco.training_physical.reshape(d_pco.training_shape)
        tcp = d_pco.predict_physical.reshape(d_pco.obs_shape)
        tc = np.concatenate((tc, tcp), axis=0)

        # Plot parameters for predictor
        xlabel = "Observation index number"
        ylabel = "Concentration ($g/m^{3})$"
        factor = 1000
        labelsize = 11

        curves(
            cols,
            tc=tc,
            sdir=sdir,
            xlabel=xlabel,
            ylabel=ylabel,
            factor=factor,
            labelsize=labelsize,
            highlight=[len(tc) - 1],
        )

        curves(
            cols,
            tc=tc,
            sdir=sdir,
            xlabel=xlabel,
            ylabel=ylabel,
            factor=factor,
            labelsize=labelsize,
            highlight=[len(tc) - 1],
            ghost=True,
            title="curves_ghost",
        )

        curves_i(
            cols,
            tc=tc,
            xlabel=xlabel,
            ylabel=ylabel,
            factor=factor,
            labelsize=labelsize,
            sdir=sdir,
            highlight=[len(tc) - 1],
        )

    if h:
        # WHP - h test + training
        fig_dir = jp(hbase, "roots_whpa")
        ff = jp(fig_dir, f"{root}.pdf")  # figure name
        h_test = np.load(jp(fig_dir, f"{root}.npy")).reshape(h_pco.obs_shape)
        h_training = h_pco.training_physical.reshape(h_pco.training_shape)
        # Plots target training + prediction
        whpa_plot(whpa=h_training, color="blue", alpha=0.5)
        whpa_plot(
            whpa=h_test,
            color="r",
            lw=2,
            alpha=0.8,
            xlabel="X(m)",
            ylabel="Y(m)",
            labelsize=11,
        )
        colors = ["blue", "red"]
        labels = ["Training", "Test"]
        legend = proxy_annotate(annotation=["C"], loc=2, fz=14)
        proxy_legend(legend1=legend, colors=colors, labels=labels, fig_file=ff)

        # WHPs
        ff = jp(md, "uq", f"{root}_cca_{cca_operator.n_components}.pdf")
        h_training = h_pco.training_physical.reshape(h_pco.training_shape)
        post_obj = joblib.load(jp(md, "obj", "post.pkl"))
        forecast_posterior = post_obj.bel_predict(
            pca_d=d_pco,
            pca_h=h_pco,
            cca_obj=cca_operator,
            n_posts=Setup.HyperParameters.n_posts,
            add_comp=False,
        )

        # I display here the prior h behind the forecasts sampled from the posterior.
        well_ids = [0] + list(map(int, list(folder)))
        labels = ["Training", "Samples", "True test"]
        colors = ["darkblue", "darkred", "k"]

        # Training
        _, well_legend = whpa_plot(
            whpa=h_training,
            alpha=0.5,
            lw=0.5,
            color=colors[0],
            show_wells=True,
            well_ids=well_ids,
            show=False,
        )

        # Samples
        whpa_plot(
            whpa=forecast_posterior,
            color=colors[1],
            lw=1,
            alpha=0.8,
            highlight=True,
            show=False,
        )

        # True test
        whpa_plot(
            whpa=h_test,
            color=colors[2],
            lw=0.8,
            alpha=1,
            x_lim=[800, 1200],
            xlabel="X(m)",
            ylabel="Y(m)",
            labelsize=11,
        )

        # Other tricky operation to add annotation
        legend_an = proxy_annotate(annotation=annotation, loc=2, fz=14)

        # Tricky operation to add a second legend:
        proxy_legend(
            legend1=well_legend,
            extra=[legend_an],
            colors=colors,
            labels=labels,
            fig_file=ff,
        )


def plot_K_field(root: str = None, wells=None, deprecated: bool = True):
    if wells is None:
        wells = Setup.Wells()

    matrix = np.load(jp(Setup.Directories.hydro_res_dir, root, "hk0.npy"))
    grid_dim = Setup.GridDimensions
    extent = (grid_dim.xo, grid_dim.x_lim, grid_dim.yo, grid_dim.y_lim)

    hkf = jp(Setup.Directories.forecasts_dir, root, "k_field.png")

    if deprecated:
        # HK field
        plt.figure()
        ax = plt.gca()
        im = ax.imshow(np.log10(matrix), cmap="coolwarm", extent=extent)
        plt.xlabel("X(m)", fontsize=11)
        plt.ylabel("Y(m)", fontsize=11)
        plot_wells(wells, markersize=3.5)
        # well_legend = plt.legend(fontsize=11, loc=2, framealpha=.6)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.ax.set_title("$Log_{10} m/s$")
        plt.savefig(hkf, bbox_inches="tight", dpi=300, transparent=True)
        plt.close()


def mode_histo(colors: list,
               an_i: int,
               wm: np.array,
               fig_name: str = "average"):
    alphabet = string.ascii_uppercase
    wid = list(map(str, Setup.Wells.combination))  # Wel identifiers (n)

    modes = []  # Get MHD corresponding to each well's mode
    for i, m in enumerate(wm):  # For each well, look up its MHD distribution
        count, values = np.histogram(m, bins="fd")
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
    plt.title("Amount of information of each well")
    plt.xlabel("Well ID")
    plt.ylabel("Opposite deviation from mode's mean")
    plt.grid(color="#95a5a6",
             linestyle="-",
             linewidth=0.5,
             axis="y",
             alpha=0.7)

    legend_a = proxy_annotate(annotation=[alphabet[an_i + 1]], loc=2, fz=14)
    plt.gca().add_artist(legend_a)

    plt.savefig(
        os.path.join(Setup.Directories.forecasts_dir,
                     f"{fig_name}_well_mode.pdf"),
        dpi=300,
        transparent=True,
    )
    plt.close()
    # plt.show()

    # Plot histogram
    for i, m in enumerate(wm):
        sns.kdeplot(m, color=f"{colors[i]}", shade=True, linewidth=2)
    plt.title("Summed MHD distribution for each well")
    plt.xlabel("Summed MHD")
    plt.ylabel("KDE")
    legend_1 = plt.legend(wid, loc=1)
    plt.gca().add_artist(legend_1)
    plt.grid(alpha=0.2)

    legend_a = proxy_annotate(annotation=[alphabet[an_i]], loc=2, fz=14)
    plt.gca().add_artist(legend_a)

    plt.savefig(
        os.path.join(Setup.Directories.forecasts_dir, f"{fig_name}_hist.pdf"),
        dpi=300,
        transparent=True,
    )
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


def curves(
    cols: list,
    tc: np.array,
    highlight=None,
    ghost=False,
    sdir=None,
    labelsize=12,
    factor=1,
    xlabel=None,
    ylabel=None,
    title="curves",
    show=False,
):
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
                plt.plot(tc[i][t] * factor,
                         color=cols[t],
                         linewidth=2,
                         alpha=1)
            elif not ghost:
                plt.plot(tc[i][t] * factor,
                         color=cols[t],
                         linewidth=0.2,
                         alpha=0.5)

    plt.grid(linewidth=0.3, alpha=0.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tick_params(labelsize=labelsize)
    if sdir:
        experiment.utils.dirmaker(sdir)
        plt.savefig(jp(sdir, f"{title}.pdf"), dpi=300, transparent=True)
        plt.close()
    if show:
        plt.show()
        plt.close()


def curves_i(
    cols,
    tc,
    highlight=None,
    labelsize=12,
    factor=1,
    xlabel=None,
    ylabel=None,
    sdir=None,
    show=False,
):
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
    title = "curves"
    n_sim, n_wels, nts = tc.shape
    for t in range(n_wels):
        for i in range(n_sim):
            if i in highlight:
                plt.plot(tc[i][t] * factor, color="k", linewidth=2, alpha=1)
            else:
                plt.plot(tc[i][t] * factor,
                         color=cols[t],
                         linewidth=0.2,
                         alpha=0.5)
        colors = [cols[t], "k"]
        plt.grid(linewidth=0.3, alpha=0.4)
        plt.tick_params(labelsize=labelsize)
        # plt.title(f'Well {t + 1}')

        alphabet = string.ascii_uppercase
        legend_a = proxy_annotate([f"{alphabet[t]}. Well {t + 1}"],
                                  fz=12,
                                  loc=2)

        labels = ["Training", "Test"]
        proxy_legend(legend1=legend_a, colors=colors, labels=labels, loc=1)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if sdir:
            experiment.utils.dirmaker(sdir)
            plt.savefig(jp(sdir, f"{title}_{t + 1}.pdf"),
                        dpi=300,
                        transparent=True)
            plt.close()
        if show:
            plt.show()
            plt.close()


def plot_wells(wells, well_ids=None, markersize: float = 4.0):
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
            label = "pw"
        else:
            label = f"{n}"
        if n in comb:
            plt.plot(
                wbd[i]["coordinates"][0],
                wbd[i]["coordinates"][1],
                f'{wbd[i]["color"]}o',
                markersize=markersize,
                markeredgecolor="k",
                markeredgewidth=0.5,
                label=label,
            )
        s += 1


def plot_head_field(root: str = None):
    matrix = np.load(
        jp(Setup.Directories.hydro_res_dir, root, "whpa_heads.npy"))
    grid_dim = Setup.GridDimensions
    extent = (grid_dim.xo, grid_dim.x_lim, grid_dim.yo, grid_dim.y_lim)

    hkf = jp(Setup.Directories.forecasts_dir, root, "heads_field.png")

    # HK field
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(matrix[0], cmap="Blues_r", extent=extent)
    plt.xlabel("X(m)", fontsize=11)
    plt.ylabel("Y(m)", fontsize=11)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.set_title("Head (m)")

    plt.savefig(hkf, bbox_inches="tight", dpi=300, transparent=True)
    plt.close()


def plot_pc_ba(root: str = None, data: bool = False, target: bool = False):
    """
    Comparison between original variables and the same variables back-transformed with n PCA components.
    :param root:
    :param data:
    :param target:
    :return:
    """

    if isinstance(root, (list, tuple)):
        if len(root) > 1:
            print("Input error")
            return
        else:
            root = root[0]

    base_dir = os.path.join(Setup.Directories.forecasts_dir, "base")
    if target:
        fobj = os.path.join(base_dir, "h_pca.pkl")
        h_pco = joblib.load(fobj)
        hnc0 = h_pco.n_pc_cut
        # mplot.h_pca_inverse_plot(h_pco, hnc0, training=True, fig_dir=os.path.join(base_dir, 'control'))

        h_pred = np.load(os.path.join(base_dir, "roots_whpa",
                                      f"{root}.npy"))  # Signed Distance
        # Cut desired number of PC components
        h_pco.test_transform(h_pred, test_root=[root])
        h_pco.comp_refresh(hnc0)
        h_pca_inverse_plot(pca_o=h_pco,
                           training=False,
                           fig_dir=jp(base_dir, "roots_whpa"))

    # d
    if data:
        subdir = os.path.join(Setup.Directories.forecasts_dir, root)
        listme = os.listdir(subdir)
        folders = list(
            filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))

        for f in folders:
            res_dir = os.path.join(subdir, f, "obj")
            # Load objects
            d_pco = joblib.load(os.path.join(res_dir, "d_pca.pkl"))
            dnc0 = d_pco.n_pc_cut  # Number of PC components
            d_pco.comp_refresh(dnc0)  # refresh based on dnc0
            setattr(d_pco, "test_root", [root])
            # mplot.d_pca_inverse_plot(d_pco, dnc0, training=True,
            #                          fig_dir=os.path.join(os.path.dirname(res_dir), 'pca'))
            # Plot parameters for predictor
            xlabel = "Observation index number"
            ylabel = "Concentration ($g/m^{3})$"
            factor = 1000
            labelsize = 11
            d_pca_inverse_plot(
                d_pco,
                xlabel=xlabel,
                ylabel=ylabel,
                labelsize=labelsize,
                factor=factor,
                training=False,
                fig_dir=os.path.join(os.path.dirname(res_dir), "pca"),
            )


def plot_whpa(root: str = None):
    """
    Loads target pickle and plots all training WHPA
    :param root:
    :return:
    """

    if isinstance(root, (list, tuple)):
        if len(root) > 1:
            print("Input error")
            return
        else:
            root = root[0]

    base_dir = os.path.join(Setup.Directories.forecasts_dir, "base")
    fobj = os.path.join(Setup.Directories.forecasts_dir, "base", "h_pca.pkl")
    h = joblib.load(fobj)
    h_training = h.training_physical.reshape(h.training_shape)

    whpa_plot(whpa=h_training,
              highlight=True,
              halpha=0.5,
              lw=0.1,
              color="darkblue",
              alpha=0.5)

    if root is not None:
        h_pred = np.load(os.path.join(base_dir, "roots_whpa", f"{root}.npy"))
        whpa_plot(
            whpa=h_pred,
            color="darkred",
            lw=1,
            alpha=1,
            annotation=["C"],
            xlabel="X(m)",
            ylabel="Y(m)",
            labelsize=11,
        )

        labels = ["Training", "Test"]
        legend = proxy_annotate(annotation=["C"], loc=2, fz=14)
        proxy_legend(
            legend1=legend,
            colors=["darkblue", "darkred"],
            labels=labels,
            fig_file=os.path.join(Setup.Directories.forecasts_dir, root,
                                  "whpa_training.pdf"),
        )


def cca_vision(root: str = None, folders: list = None):
    """
    Loads CCA pickles and plots components for all folders
    :param root:
    :param folders:
    :return:
    """

    if isinstance(root, (list, tuple)):
        if len(root) > 1:
            print("Input error")
            return
        else:
            root = root[0]

    subdir = os.path.join(Setup.Directories.forecasts_dir, root)

    if folders is None:
        listme = os.listdir(subdir)
        folders = list(
            filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))
    else:
        if not isinstance(folders, (list, tuple)):
            folders = [folders]
        else:
            pass

    for f in folders:
        res_dir = os.path.join(subdir, f, "obj")

        # Load objects
        d_cca_training, h_cca_training, *rest = reload_trained_model(root=root,
                                                                     well=f)

        kde_cca(root=root, well=f, sdir=os.path.join(subdir, f, "cca"))

        # CCA coefficient plot
        cca_coefficient = np.corrcoef(
            d_cca_training,
            h_cca_training,
        ).diagonal(offset=d_cca_training.shape[0])
        plt.plot(cca_coefficient, "lightblue", zorder=1)
        plt.scatter(
            x=np.arange(len(cca_coefficient)),
            y=cca_coefficient,
            c=cca_coefficient,
            alpha=1,
            s=50,
            cmap="coolwarm",
            zorder=2,
        )
        cb = plt.colorbar()
        cb.ax.set_title("R")
        plt.grid(alpha=0.4, linewidth=0.5, zorder=0)
        plt.xticks(np.arange(len(cca_coefficient)),
                   np.arange(1,
                             len(cca_coefficient) + 1))
        plt.tick_params(labelsize=5)
        plt.yticks([])
        # plt.title('Decrease of CCA correlation coefficient with component number')
        plt.ylabel("Correlation coefficient")
        plt.xlabel("Component number")

        # Add annotation
        legend = proxy_annotate(annotation=["D"], fz=14, loc=1)
        plt.gca().add_artist(legend)

        plt.savefig(
            os.path.join(os.path.dirname(res_dir), "cca", "coefs.pdf"),
            bbox_inches="tight",
            dpi=300,
            transparent=True,
        )
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
            print("Input error")
            return
        else:
            root = root[0]

    subdir = os.path.join(Setup.Directories.forecasts_dir, root)
    if folders is None:
        listme = os.listdir(subdir)
        folders = list(
            filter(lambda du: os.path.isdir(os.path.join(subdir, du)), listme))
    else:
        if not isinstance(folders, (list, tuple)):
            folders = [folders]

    if d:
        for f in folders:
            dfig = os.path.join(subdir, f, "pca")
            # For d only
            pcaf = os.path.join(subdir, f, "obj", "d_pca.pkl")
            d_pco = joblib.load(pcaf)
            fig_file = os.path.join(dfig, "d_scores.pdf")
            if scores:
                pca_scores(
                    training=d_pco.training_pc,
                    prediction=d_pco.predict_pc,
                    n_comp=d_pco.n_pc_cut,
                    annotation=["E"],
                    labels=labels,
                    fig_file=fig_file,
                )
            # Explained variance plots
            if exvar:
                fig_file = os.path.join(dfig, "d_exvar.pdf")
                explained_variance(
                    d_pco.operator,
                    n_comp=d_pco.n_pc_cut,
                    thr=0.8,
                    annotation=["C"],
                    fig_file=fig_file,
                )
    if h:
        hbase = os.path.join(Setup.Directories.forecasts_dir, "base")
        # Load h pickle
        pcaf = os.path.join(hbase, "h_pca.pkl")
        h_pco = joblib.load(pcaf)
        # Load npy whpa prediction
        prediction = np.load(os.path.join(hbase, "roots_whpa", f"{root}.npy"))
        # Transform and split
        h_pco.test_transform(prediction, test_root=[root])
        nho = h_pco.n_pc_cut
        h_pc_training, h_pc_prediction = h_pco.comp_refresh(nho)
        # Plot
        fig_file = os.path.join(hbase, "roots_whpa", f"{root}_pca_scores.pdf")
        if scores:
            pca_scores(
                training=h_pc_training,
                prediction=h_pc_prediction,
                n_comp=nho,
                annotation=["F"],
                labels=labels,
                fig_file=fig_file,
            )
        # Explained variance plots
        if exvar:
            fig_file = os.path.join(hbase, "roots_whpa",
                                    f"{root}_pca_exvar.pdf")
            explained_variance(
                h_pco.operator,
                n_comp=h_pco.n_pc_cut,
                thr=0.85,
                annotation=["D"],
                fig_file=fig_file,
            )


def check_root(xlim: list, ylim: list, root: list):
    """
    Plots raw data of folder 'root'
    :param xlim:
    :param ylim:
    :param root:
    :return:
    """
    bkt, whpa, _ = experiment.utils.data_loader(roots=root)
    whpa = whpa.squeeze()
    # self.curves_i(bkt, show=True)  # This function will not work

    plt.plot(whpa[:, 0], whpa[:, 1], "wo")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def d_pca_inverse_plot(
    pca_o,
    factor: float = 1.0,
    xlabel: str = None,
    ylabel: str = None,
    labelsize: float = 11.0,
    training=True,
    fig_dir=None,
    show=False,
):
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

        plt.plot(to_plot * factor, "r", alpha=0.8)
        plt.plot(v_pred * factor, "b", alpha=0.8)
        # Add title inside the box
        an = ["A"]
        legend_a = proxy_annotate(annotation=an, loc=2, fz=14)
        proxy_legend(
            legend1=legend_a,
            colors=["red", "blue"],
            labels=["Physical", "Back transformed"],
            marker=["-", "-"],
            loc=1,
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tick_params(labelsize=labelsize)

        # Increase y axis by a small percentage for annotation in upper left corner
        yrange = np.max(to_plot * factor) * 1.15
        plt.ylim([0, yrange])

        if fig_dir is not None:
            experiment.utils.dirmaker(fig_dir)
            plt.savefig(jp(fig_dir, f"{r}_d.pdf"), dpi=300, transparent=True)
            plt.close()
        if show:
            plt.show()
            plt.close()


def hydro_examination(root: str):
    md = Setup.Directories()
    ep = jp(md.hydro_res_dir, root, "tracking_ep.npy")
    epxy = np.load(ep)

    plt.plot(epxy[:, 0], epxy[:, 1], "ko")
    # seed = np.random.randint(2**32 - 1)
    # np.random.seed(seed)
    # sample = np.random.randint(144, size=10)
    sample = np.array([94, 10, 101, 29, 43, 116, 100, 40, 72])
    for i in sample:
        plt.text(
            epxy[i, 0] + 4,
            epxy[i, 1] + 4,
            i,
            color="black",
            fontsize=11,
            weight="bold",
            bbox=dict(facecolor="white",
                      edgecolor="black",
                      boxstyle="round,pad=.5",
                      alpha=0.7),
        )
        # plt.annotate(i, (epxy[i, 0] + 4, epxy[i, 1] + 4), fontsize=14, weight='bold', color='r')

    plt.grid(alpha=0.5)
    plt.xlim([870, 1080])
    plt.ylim([415, 600])
    plt.xlabel("X(m)")
    plt.ylabel("Y(m)")
    plt.tick_params(labelsize=11)

    legend = proxy_annotate(annotation=["A"], loc=2, fz=14)
    plt.gca().add_artist(legend)

    plt.savefig(
        jp(md.forecasts_dir, "base", "roots_whpa", f"{root}_ep.pdf"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()


def despine(
    fig=None,
    ax=None,
    top=True,
    right=True,
    left=False,
    bottom=False,
    offset=None,
    trim=False,
):
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
        axes_dummy = plt.gcf().axes
    elif fig is not None:
        axes_dummy = fig.axes
    elif ax is not None:
        axes_dummy = [ax]
    else:
        return

    for ax_i in axes_dummy:
        for side in ["top", "right", "left", "bottom"]:
            # Toggle the spine objects
            is_visible = not locals()[side]
            ax_i.spines[side].set_visible(is_visible)
            if offset is not None and is_visible:
                try:
                    val = offset.get(side, 0)
                except AttributeError:
                    val = offset
                ax_i.spines[side].set_position(("outward", val))

        # Potentially move the ticks
        if left and not right:
            maj_on = any(t.tick1line.get_visible()
                         for t in ax_i.yaxis.majorTicks)
            min_on = any(t.tick1line.get_visible()
                         for t in ax_i.yaxis.minorTicks)
            ax_i.yaxis.set_ticks_position("right")
            for t in ax_i.yaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.yaxis.minorTicks:
                t.tick2line.set_visible(min_on)

        if bottom and not top:
            maj_on = any(t.tick1line.get_visible()
                         for t in ax_i.xaxis.majorTicks)
            min_on = any(t.tick1line.get_visible()
                         for t in ax_i.xaxis.minorTicks)
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
                ax_i.spines["bottom"].set_bounds(firsttick, lasttick)
                ax_i.spines["top"].set_bounds(firsttick, lasttick)
                newticks = xticks.compress(xticks <= lasttick)
                newticks = newticks.compress(newticks >= firsttick)
                ax_i.set_xticks(newticks)

            yticks = np.asarray(ax_i.get_yticks())
            if yticks.size:
                firsttick = np.compress(yticks >= min(ax_i.get_ylim()),
                                        yticks)[0]
                lasttick = np.compress(yticks <= max(ax_i.get_ylim()),
                                       yticks)[-1]
                ax_i.spines["left"].set_bounds(firsttick, lasttick)
                ax_i.spines["right"].set_bounds(firsttick, lasttick)
                newticks = yticks.compress(yticks <= lasttick)
                newticks = newticks.compress(newticks >= firsttick)
                ax_i.set_yticks(newticks)


def order_vertices(vertices):
    """
    Paraview expects vertices in a particular order, with the origin at the bottom left corner.
    :param vertices: (x, y) coordinates of the polygon vertices
    :return:
    """
    # Compute center of vertices
    center = tuple(
        map(
            operator.truediv,
            reduce(lambda x, y: map(operator.add, x, y), vertices),
            [len(vertices)] * 2,
        ))

    # Sort vertices according to angle
    so = sorted(
        vertices,
        key=lambda coord: (math.degrees(
            math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))),
    )

    return np.array(so)


class ModelVTK:
    """
    Loads flow/transport models and export the VTK objects.
    """

    def __init__(self, base=None, folder=None):
        self.base = base
        md = self.base.Directories()
        self.rn = folder
        self.bdir = md.main_dir
        self.results_dir = jp(md.hydro_res_dir, self.rn)
        self.vtk_dir = jp(self.results_dir, "vtk")
        experiment.utils.dirmaker(self.vtk_dir)

        # Load flow model
        try:
            m_load = jp(self.results_dir, "whpa.nam")
            self.flow_model = experiment.utils.load_flow_model(
                m_load, model_ws=self.results_dir)
            delr = self.flow_model.modelgrid.delr  # thicknesses along rows
            delc = self.flow_model.modelgrid.delc  # thicknesses along column
            # xyz_vertices = self.flow_model.modelgrid.xyzvertices
            # blocks2d = mops.blocks_from_rc(delc, delr)
            self.blocks = experiment.algorithms.spatial.blocks_from_rc_3d(
                delc, delr)
            # blocks3d = self.blocks.reshape(-1, 3)
        except Exception as e:
            print(e)

        try:
            # Transport model
            mt_load = jp(self.results_dir, "whpa.mtnam")
            self.transport_model = experiment.utils.load_transport_model(
                mt_load, self.flow_model, model_ws=self.results_dir)
            ucn_files = [
                jp(self.results_dir, f"MT3D00{i}.UCN")
                for i in Setup.Wells.combination
            ]  # Files containing concentration
            ucn_obj = [flopy.utils.UcnFile(uf)
                       for uf in ucn_files]  # Load them
            self.times = [uo.get_times() for uo in ucn_obj]  # Get time steps
            self.concs = np.array([uo.get_alldata()
                                   for uo in ucn_obj])  # Get all data
        except Exception as e:
            print(e)

    def flow_vtk(self):
        """
        Export flow model packages and computed heads to vtk format

        """
        dir_fv = jp(self.results_dir, "vtk", "flow")
        experiment.utils.dirmaker(dir_fv)
        self.flow_model.export(dir_fv, fmt="vtk")
        vtk_flow.export_heads(
            self.flow_model,
            jp(self.results_dir, "whpa.hds"),
            jp(self.results_dir, "vtk", "flow"),
            binary=True,
            kstpkper=(0, 0),
        )

    # Load transport model

    def transport_vtk(self):
        """
        Export transport package attributes to vtk format

        """
        dir_tv = jp(self.results_dir, "vtk", "transport")
        experiment.utils.dirmaker(dir_tv)
        self.transport_model.export(dir_tv, fmt="vtk")

    # Export UCN to vtk

    def stacked_conc_vtk(self):
        """Stack component concentrations for each time step and save vtk"""
        conc_dir = jp(self.results_dir, "vtk", "transport", "concentration")
        experiment.utils.dirmaker(conc_dir)
        # First replace 1e+30 value (inactive cells) by 0.
        conc0 = np.abs(np.where(self.concs == 1e30, 0, self.concs))
        # Cells configuration for 3D blocks
        cells = [("quad", np.array([list(np.arange(i * 4, i * 4 + 4))]))
                 for i in range(len(self.blocks))]
        for j in range(self.concs.shape[1]):
            # Stack the concentration of each component at each time step to visualize them in one plot.
            array = np.zeros(self.blocks.shape[0])
            dic_conc = {}
            for i in range(1, 7):
                # fliplr is necessary as the reshape modifies the array structure
                cflip = np.fliplr(conc0[i - 1, j]).reshape(-1)
                dic_conc["conc_wel_{}".format(i)] = cflip
                array += cflip  # Stack all components
            dic_conc["stack"] = array

    def conc_vtk(self):
        """Stack component concentrations for each time step and save vtk"""
        # First replace 1e+30 value (inactive cells) by 0.
        conc0 = np.abs(np.where(self.concs == 1e30, 0, self.concs))

        # Initiate points and ugrid
        points = vtk.vtkPoints()
        ugrid = vtk.vtkUnstructuredGrid()

        for e, b in enumerate(self.blocks):
            # Order vertices in vtkPixel convention
            sb = sorted(b, key=lambda k: [k[1], k[0]])
            [points.InsertPoint(e * 4 + es, bb) for es, bb in enumerate(sb)
             ]  # Insert points by giving first their index e*4+es
            ugrid.InsertNextCell(vtk.VTK_PIXEL, 4, list(range(
                e * 4, e * 4 + 4)))  # Insert cell in UGrid

        ugrid.SetPoints(points)  # Set points

        for i in range(1, 7):  # For eaxh injecting well
            conc_dir = jp(self.results_dir, "vtk", "transport",
                          "{}_UCN".format(i))
            experiment.utils.dirmaker(conc_dir)
            # Initiate array and give it a name
            concArray = vtk.vtkDoubleArray()
            concArray.SetName(f"conc{i}")
            for j in range(self.concs.shape[1]):  # For each time step
                # Set array
                array = np.fliplr(conc0[i - 1, j]).reshape(-1)
                [concArray.InsertNextValue(s) for s in array]
                ugrid.GetCellData().AddArray(
                    concArray)  # Add array to unstructured grid

                # Save grid
                writer = vtk.vtkXMLUnstructuredGridWriter()
                writer.SetInputData(ugrid)
                writer.SetFileName(jp(conc_dir, "{}_conc.vtu".format(j)))
                writer.Write()

                # Clear storage but keep name
                concArray.Initialize()
                ugrid.GetCellData().Initialize()

    # %% Plot modpath

    def particles_vtk(self, path=1):
        """
        Export travelling particles time series in VTP format
        :param path: Flag to export path's vtk
        :return:
        """
        back_dir = jp(self.results_dir, "vtk", "backtrack")
        experiment.utils.dirmaker(back_dir)

        # mp_reloaded = backtrack(flowmodel=flow_model, exe_name='', load=True)
        # load the endpoint data
        # endfile = jp(results_dir, 'whpa_mp.mpend')
        # endobj = flopy.utils.EndpointFile(endfile)
        # ept = endobj.get_alldata()

        # # load the pathline data
        # The same information is stored in the time series file
        # pthfile = jp(results_dir, 'whpa_mp.mppth')
        # pthobj = flopy.utils.PathlineFile(pthfile)
        # plines = pthobj.get_alldata()

        # load the time series
        tsfile = jp(self.results_dir, "whpa_mp.timeseries")
        tso = flopy.utils.modpathfile.TimeseriesFile(tsfile)
        ts = tso.get_alldata()

        n_particles = len(ts)
        n_t_stp = ts[0].shape[0]
        time_steps = ts[0].time

        points_x = np.array([ts[i].x for i in range(len(ts))])
        points_y = np.array([ts[i].y for i in range(len(ts))])
        points_z = np.array([ts[i].z for i in range(len(ts))])

        xs = points_x[:, 0]  # Data at first time step
        ys = points_y[:, 0]
        # Replace elevation by 0 to project them in the surface
        zs = points_z[:, 0] * 0
        prev = np.vstack((xs, ys, zs)).T.reshape(-1, 3)

        speed_array = None

        for i in range(n_t_stp):  # For each time step i
            # Get all particles positions
            xs = points_x[:, i]
            ys = points_y[:, i]
            # Replace elevation by 0 to project them in the surface
            zs = points_z[:, i] * 0
            xyz_particles_t_i = np.vstack((xs, ys, zs)).T.reshape(-1, 3)

            # Compute instant speed ds/dt
            speed = np.abs(
                operator.truediv(
                    tuple(map(np.linalg.norm, xyz_particles_t_i - prev)),
                    time_steps[i] - time_steps[i - 1],
                ))
            prev = xyz_particles_t_i

            if speed_array is None:
                speed_array = np.array([speed])
            else:
                speed_array = np.append(speed_array, np.array([speed]), 0)

            # Initiate points object
            points = vtk.vtkPoints()
            ids = [points.InsertNextPoint(c) for c in xyz_particles_t_i]

            # Create a cell array to store the points
            vertices = vtk.vtkCellArray()
            vertices.InsertNextCell(n_particles)
            [vertices.InsertCellPoint(ix) for ix in ids]

            # Create a polydata to store everything in
            polyData = vtk.vtkPolyData()
            # Add the points to the dataset
            polyData.SetPoints(points)
            polyData.SetVerts(vertices)

            # Assign value array
            speedArray = vtk.vtkDoubleArray()
            speedArray.SetName("Speed")
            [speedArray.InsertNextValue(s) for s in speed]
            polyData.GetPointData().AddArray(speedArray)

            # Write data
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetInputData(polyData)
            writer.SetFileName(jp(back_dir, "particles_t{}.vtp".format(i)))
            writer.Write()

            # Write path lines
            if i and path:
                for p in range(n_particles):
                    short = np.vstack((
                        points_x[p, :i + 1],
                        points_y[p, :i + 1],
                        np.abs(points_z[p, :i + 1] * 0),
                    )).T
                    points = vtk.vtkPoints()
                    [points.InsertNextPoint(c) for c in short]

                    # Create a polydata to store everything in
                    polyData = vtk.vtkPolyData()
                    # Add the points to the dataset
                    polyData.SetPoints(points)

                    # Create a cell array to store the lines in and add the lines to it
                    cells = vtk.vtkCellArray()
                    cells.InsertNextCell(i + 1)
                    [cells.InsertCellPoint(k) for k in range(i + 1)]

                    # Create value array and assign it to the polydata
                    speed = vtk.vtkDoubleArray()
                    speed.SetName("speed")
                    [
                        speed.InsertNextValue(speed_array[k][p])
                        for k in range(i + 1)
                    ]
                    polyData.GetPointData().AddArray(speed)

                    # Add the lines to the dataset
                    polyData.SetLines(cells)

                    # Export
                    writer = vtk.vtkXMLPolyDataWriter()
                    writer.SetInputData(polyData)
                    writer.SetFileName(
                        jp(back_dir, "path{}_t{}.vtp".format(p, i)))
                    writer.Write()
            else:
                for p in range(n_particles):
                    xs = points_x[p, i]
                    ys = points_y[p, i]
                    # Replace elevation by 0 to project them in the surface
                    zs = points_z[p, i] * 0
                    xyz_particles_t_i = np.vstack(
                        (xs, ys, zs)).T.reshape(-1, 3)

                    # Create points
                    points = vtk.vtkPoints()
                    ids = [
                        points.InsertNextPoint(c) for c in xyz_particles_t_i
                    ]

                    # Create a cell array to store the points
                    vertices = vtk.vtkCellArray()
                    # Why is this line necessary ?
                    vertices.InsertNextCell(len(xyz_particles_t_i))
                    [vertices.InsertCellPoint(ix) for ix in ids]

                    # Create a polydata to store everything in
                    polyData = vtk.vtkPolyData()
                    # Add the points to the dataset
                    polyData.SetPoints(points)
                    polyData.SetVerts(vertices)

                    # Export
                    writer = vtk.vtkXMLPolyDataWriter()
                    writer.SetInputData(polyData)
                    writer.SetFileName(
                        jp(back_dir, "path{}_t{}.vtp".format(p, i)))
                    writer.Write()

    # %% Export wells objects as vtk

    def wells_vtk(self):
        """Exports wells coordinates to VTK"""

        wbd = self.base.Wells().wells_data

        wels = np.array([wbd[o]["coordinates"] for o in wbd])
        wels = np.insert(wels, 2, np.zeros(len(wels)),
                         axis=1)  # Insert zero array for Z

        # Export wells as VTK points
        points = vtk.vtkPoints()  # Points
        ids = [points.InsertNextPoint(w) for w in wels]  # Points IDS
        welArray = vtk.vtkCellArray()  # Vertices
        welArray.InsertNextCell(len(wels))
        [welArray.InsertCellPoint(ix) for ix in ids]
        welPolydata = vtk.vtkPolyData()  # PolyData to store everything
        welPolydata.SetPoints(points)
        welPolydata.SetVerts(welArray)
        # Save objects
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(welPolydata)
        writer.SetFileName(jp(self.vtk_dir, "wells.vtp"))
        writer.Write()


def get_defaults_kde_plot():
    height = 6
    ratio = 6
    space = 0

    xlim = None
    ylim = None
    marginal_ticks = False

    # Set up the subplot grid
    f = plt.figure(figsize=(height, height))
    gs = plt.GridSpec(ratio + 1, ratio + 1)

    # ax_joint = f.add_subplot(gs[1:, :-1])
    # ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
    # ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)

    ax_joint = f.add_subplot(gs[1:, 1:-1])
    ax_marg_x = f.add_subplot(gs[0, 1:-1], sharex=ax_joint)
    ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)
    ax_cb = f.add_subplot(gs[1:, 0])

    fig = f
    ax_joint = ax_joint
    ax_marg_x = ax_marg_x
    ax_marg_y = ax_marg_y
    ax_cb = ax_cb

    # Turn off tick visibility for the measure axis on the marginal plots
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    plt.setp(ax_marg_x.get_xticklabels(minor=True), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(minor=True), visible=False)

    # Turn off the ticks on the density axis for the marginal plots
    plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)

    plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)

    plt.setp(ax_marg_x.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)

    plt.setp(ax_marg_x.get_yticklabels(minor=True), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(minor=True), visible=False)

    ax_marg_x.yaxis.grid(False)
    ax_marg_y.xaxis.grid(False)

    ax_cb.axis("off")

    if xlim is not None:
        ax_joint.set_xlim(xlim)
    if ylim is not None:
        ax_joint.set_ylim(ylim)

    # Make the grid look nice
    despine(f)
    if not marginal_ticks:
        despine(ax=ax_marg_x, left=True)
        despine(ax=ax_marg_y, bottom=True)

    for axes in [ax_marg_x, ax_marg_y, ax_cb]:
        for axis in [axes.xaxis, axes.yaxis]:
            axis.label.set_visible(False)
    f.tight_layout()
    f.subplots_adjust(hspace=space, wspace=space)

    return ax_joint, ax_marg_x, ax_marg_y, ax_cb


def kde_cca(
    root: str,
    well: str,
    sample_n: int = 0,
    sdir: str = None,
    show: bool = False,
    dist_plot: bool = False,
):
    # Reload model
    d, h, d_cca_prediction, h_cca_prediction, post, cca_operator = reload_trained_model(
        root=root, well=well, sample_n=sample_n)
    # Find max kde value
    vmax = 0
    for comp_n in range(cca_operator.n_components):

        hp, sup = stats.posterior_conditional(x=d[comp_n],
                                              y=h[comp_n],
                                              x_obs=d_cca_prediction[comp_n])

        # Plot h posterior given d
        density, _ = stats.kde_params(x=d[comp_n], y=h[comp_n])
        maxloc = np.max(density)
        if vmax < maxloc:
            vmax = maxloc

    cca_coefficient = np.corrcoef(d, h).diagonal(
        offset=cca_operator.n_components)  # Gets correlation coefficient

    for comp_n in range(cca_operator.n_components):
        # Get figure default parameters
        ax_joint, ax_marg_x, ax_marg_y, ax_cb = get_defaults_kde_plot()

        # Conditional:
        hp, sup = stats.posterior_conditional(x=d[comp_n],
                                              y=h[comp_n],
                                              x_obs=d_cca_prediction[comp_n])

        # load prediction object
        post_test = post.random_sample(Setup.HyperParameters.n_posts).T
        post_test_t = post.normalize_h.transform(post_test.T).T
        y_samp = post_test_t[comp_n]

        # Plot h posterior given d
        density, support = stats.kde_params(x=d[comp_n],
                                            y=h[comp_n],
                                            gridsize=1000)
        xx, yy = support

        marginal_eval_x = stats.KDE()
        marginal_eval_y = stats.KDE()

        # support is cached
        kde_x, sup_x = marginal_eval_x(d[comp_n])
        kde_y, sup_y = marginal_eval_y(h[comp_n])
        # use the same support as y
        kde_y_samp, sup_samp = marginal_eval_y(y_samp)

        # Filled contour plot
        # Mask values under threshold
        z = ma.masked_where(density <= np.finfo(np.float16).eps, density)
        # Filled contour plot
        # 'BuPu_r' is nice
        cf = ax_joint.contourf(xx,
                               yy,
                               z,
                               cmap="coolwarm",
                               levels=100,
                               vmin=0,
                               vmax=vmax)
        cb = plt.colorbar(cf, ax=[ax_cb], location="left")
        cb.ax.set_title("$KDE_{Gaussian}$", fontsize=10)
        # Vertical line
        ax_joint.axvline(
            x=d_cca_prediction[comp_n],
            color="red",
            linewidth=1,
            alpha=0.5,
            label="$d^{c}_{True}$",
        )
        # Horizontal line
        ax_joint.axhline(
            y=h_cca_prediction[comp_n],
            color="deepskyblue",
            linewidth=1,
            alpha=0.5,
            label="$h^{c}_{True}$",
        )
        # Scatter plot
        ax_joint.plot(
            d[comp_n],
            h[comp_n],
            "ko",
            markersize=2,
            markeredgecolor="w",
            markeredgewidth=0.2,
            alpha=0.7,
        )
        # Point
        ax_joint.plot(
            d_cca_prediction[comp_n],
            h_cca_prediction[comp_n],
            "wo",
            markersize=5,
            markeredgecolor="k",
            alpha=1,
        )
        # Marginal x plot
        #  - Line plot
        ax_marg_x.plot(sup_x, kde_x, color="black", linewidth=0.5, alpha=1)
        #  - Fill to axis
        ax_marg_x.fill_between(sup_x,
                               0,
                               kde_x,
                               color="royalblue",
                               alpha=0.5,
                               label="$p(d^{c})$")
        #  - Notch indicating true value
        ax_marg_x.axvline(x=d_cca_prediction[comp_n],
                          ymax=0.25,
                          color="red",
                          linewidth=1,
                          alpha=0.5)
        ax_marg_x.legend(loc=2, fontsize=10)

        # Marginal y plot
        #  - Line plot
        ax_marg_y.plot(kde_y, sup_y, color="black", linewidth=0.5, alpha=1)
        #  - Fill to axis
        ax_marg_y.fill_betweenx(sup_y,
                                0,
                                kde_y,
                                alpha=0.5,
                                color="darkred",
                                label="$p(h^{c})$")
        #  - Notch indicating true value
        ax_marg_y.axhline(
            y=h_cca_prediction[comp_n],
            xmax=0.25,
            color="deepskyblue",
            linewidth=1,
            alpha=0.5,
        )
        # Marginal y plot with BEL
        #  - Line plot
        ax_marg_y.plot(kde_y_samp,
                       sup_samp,
                       color="black",
                       linewidth=0.5,
                       alpha=0)
        #  - Fill to axis
        ax_marg_y.fill_betweenx(
            sup_samp,
            0,
            kde_y_samp,
            color="teal",
            alpha=0.5,
            label="$p(h^{c}|d^{c}_{*})_{BEL}$",
        )
        # Conditional distribution
        #  - Line plot
        ax_marg_y.plot(hp, sup, color="red", alpha=0)
        #  - Fill to axis
        ax_marg_y.fill_betweenx(
            sup,
            0,
            hp,
            color="mediumorchid",
            alpha=0.5,
            label="$p(h^{c}|d^{c}_{*})_{KDE}$",
        )
        ax_marg_y.legend(fontsize=10)
        # Labels
        ax_joint.set_xlabel("$d^{c}$", fontsize=14)
        ax_joint.set_ylabel("$h^{c}$", fontsize=14)
        plt.tick_params(labelsize=14)

        # Add custom artists
        subtitle = my_alphabet(comp_n)
        # Add title inside the box
        an = [
            f"{subtitle}. Pair {comp_n + 1} - R = {round(cca_coefficient[comp_n], 3)}"
        ]
        legend_a = proxy_annotate(obj=ax_joint, annotation=an, loc=2, fz=12)
        #
        proxy_legend(
            obj=ax_joint,
            legend1=legend_a,
            colors=["black", "white", "red", "deepskyblue"],
            labels=["$Training$", "$Test$", "$d^{c}_{*}$", "$h^{c}_{True}$"],
            marker=["o", "o", "-", "-"],
            pec=["k", "k", None, None],
            fz=10,
        )

        if sdir:
            experiment.utils.dirmaker(sdir)
            plt.savefig(
                jp(sdir, f"cca_kde_{comp_n}.pdf"),
                bbox_inches="tight",
                dpi=300,
                transparent=True,
            )
            plt.close()
        if show:
            plt.show()
            plt.close()

        def posterior_distribution():
            # prior
            plt.plot(sup_y, kde_y, color="black", linewidth=0.5, alpha=1)
            plt.fill_between(sup_y,
                             0,
                             kde_y,
                             color="mistyrose",
                             alpha=1,
                             label="$p(h^{c})$")
            # posterior kde
            plt.plot(sup, hp, color="darkred", linewidth=0.5, alpha=0)
            plt.fill_between(
                sup,
                0,
                hp,
                color="salmon",
                alpha=0.5,
                label="$p(h^{c}|d^{c}_{*})$ (KDE)",
            )
            # posterior bel
            plt.plot(sup_samp,
                     kde_y_samp,
                     color="black",
                     linewidth=0.5,
                     alpha=1)
            plt.fill_between(
                sup_samp,
                0,
                kde_y_samp,
                color="gray",
                alpha=0.5,
                label="$p(h^{c}|d^{c}_{*})$ (BEL)",
            )

            # True prediction
            plt.axvline(
                x=h_cca_prediction[0],
                linewidth=3,
                alpha=0.4,
                color="deepskyblue",
                label="$h^{c}_{True}$",
            )

            # Grid
            plt.grid(alpha=0.2)

            # Tuning
            plt.ylabel("Density", fontsize=14)
            plt.xlabel("$h^{c}$", fontsize=14)
            plt.xlim([np.min(h[comp_n]), np.max(d[comp_n])])
            plt.tick_params(labelsize=14)

            plt.legend(loc=2)

            if sdir:
                experiment.utils.dirmaker(sdir)
                plt.savefig(
                    jp(sdir, f"cca_prior_post_{comp_n}.pdf"),
                    bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                )
                plt.close()
            if show:
                plt.show()
                plt.close()

        if dist_plot:
            posterior_distribution()
