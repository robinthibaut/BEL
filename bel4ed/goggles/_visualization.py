#  Copyright (c) 2021. Robin Thibaut, Ghent University
import math
import operator
import os
import string
from functools import reduce
from os.path import dirname
from os.path import join as jp

import flopy
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import vtk
from flopy.export import vtk as vtk_flow
from loguru import logger
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import BSpline, make_interp_spline
from skbel.goggles import _kde_cca, _proxy_annotate, _proxy_legend, explained_variance
from skbel.spatial import (
    blocks_from_rc_3d,
    contours_vertices,
    grid_parameters,
    refine_machine,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, deprecated

import bel4ed.utils
from bel4ed.config import Setup
from bel4ed.datasets import data_loader, load_flow_model, load_transport_model
from bel4ed.spatial import binary_stack
from bel4ed.utils import Root, reload_trained_model

__all__ = [
    "plot_results",
    "plot_K_field",
    "plot_head_field",
    "plot_whpa",
    "whpa_plot",
    "cca_vision",
    "pca_vision",
    "plot_pc_ba",
    "mode_histo",
    "ModelVTK",
]


def vertices_vtp(folder, vertices):
    vdir = folder
    bel4ed.utils.dirmaker(vdir)
    for i, v in enumerate(vertices):
        nv = len(v)
        points = vtk.vtkPoints()
        [points.InsertNextPoint(np.insert(c, 2, 0)) for c in v]
        # Create a polydata to store everything in
        poly_data = vtk.vtkPolyData()
        # Add the points to the dataset
        poly_data.SetPoints(points)
        # Create a cell array to store the lines in and add the lines to it
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(nv)
        [cells.InsertCellPoint(k) for k in range(nv)]
        # Add the lines to the dataset
        poly_data.SetLines(cells)
        # Export
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(poly_data)

        writer.SetFileName(jp(vdir, f"forecast_posterior_{i}.vtp"))
        writer.Write()


def pca_scores(
    training: np.array,
    prediction: np.array,
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
    plt.xticks(
        np.concatenate([np.array([0]), np.arange(4, n_comp, 5)]),
        np.concatenate([np.array([1]), np.arange(5, n_comp + 5, 5)]),
    )  # Plot all training scores
    plt.plot(training.T[:n_comp], "ob", markersize=3, alpha=0.1)
    # plt.plot(training.T[:ut], '+w', markersize=.5, alpha=0.2)

    # For each sample used for prediction:
    for sample_n in range(len(prediction)):
        # Select observation
        pc_obs = prediction[sample_n]
        # Create beautiful spline to follow prediction scores
        xnew = np.linspace(1, n_comp, 200)  # New points for plotting curve
        spl = make_interp_spline(
            np.arange(1, n_comp + 1), pc_obs.T[:n_comp], k=3
        )  # type: BSpline
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
    legend_a = _proxy_annotate(annotation=annotation, loc=2, fz=14)
    _proxy_legend(
        legend1=legend_a,
        colors=["blue", "red"],
        labels=["Training", "Test"],
        marker=["o", "o"],
    )

    if fig_file:
        bel4ed.utils.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, dpi=300, transparent=False)
        plt.close()
    if show:
        plt.show()


def whpa_plot(
    grf: float = None,
    well_comb: list = None,
    whpa: np.array = None,
    load_bstack: bool = False,
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
    grid: bool = True,
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

    :param grid:
    :param grf: Grid cell size
    :param well_comb: List of well combination
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
    focus = Setup.Focus
    wells = Setup.Wells

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
            extent=np.concatenate([focus.x_range, focus.y_range]),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        cb = plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.set_title(cb_title)

    if halpha is None:
        halpha = alpha

    # Plot results
    if whpa is None:
        whpa = np.array([])

    if whpa.ndim > 2:  # New approach is to plot filled contours

        new_grf = 1  # Refine grid
        _, _, new_x, new_y = refine_machine(xlim, ylim, new_grf=new_grf)

        if load_bstack:
            try:
                b_low = np.load(
                    jp(dirname(os.path.abspath(__file__)), "_temp", "blow.npy")
                )
            except FileNotFoundError:
                xys, nrow, ncol = grid_parameters(x_lim=xlim, y_lim=ylim, grf=new_grf)
                vertices = contours_vertices(x=x, y=y, arrays=whpa)
                b_low = binary_stack(xys=xys, nrow=nrow, ncol=ncol, vertices=vertices)
                np.save(
                    jp(dirname(os.path.abspath(__file__)), "_temp", "blow.npy"), b_low
                )
        else:
            xys, nrow, ncol = grid_parameters(x_lim=xlim, y_lim=ylim, grf=new_grf)
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
                contour = plt.contour(
                    z,
                    [0],
                    extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                    colors=color,
                    linewidths=lw,
                    alpha=halpha,
                )

    elif whpa.size > 0:  # If only one WHPA to display
        contour = plt.contour(
            whpa,
            [0],
            extent=np.concatenate([focus.x_range, focus.y_range]),
            colors=color,
            linewidths=lw,
            alpha=halpha,
        )
    else:
        contour = None

    # Grid
    if grid:
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
    plt.tick_params(labelsize=labelsize)

    if annotation:
        legend = _proxy_annotate(annotation=annotation, fz=14, loc=2)
        plt.gca().add_artist(legend)

    if fig_file:
        bel4ed.utils.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, bbox_inches="tight", dpi=300, transparent=False)
        plt.close()
    if show:
        plt.show()
        plt.close()

    return contour, well_legend


@deprecated()
def post_examination(
    root: str, xlim: list = None, ylim: list = None, show: bool = False
):
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

    plt.savefig(
        jp(sdir, f"{root}_SD.png"), dpi=300, bbox_inches="tight", transparent=False
    )
    if show:
        plt.show()
    plt.close()


def h_pca_inverse_plot(bel, Y_obs, fig_dir: str = None, show: bool = False):
    """
    Plot used to compare the reproduction of the original physical space after PCA transformation
    :param bel
    :param fig_dir: str:
    :param show: bool:
    :return:
    """

    shape = bel.Y_shape

    if bel.Y_obs_pc is not None:
        v_pc = check_array(bel.Y_obs_pc.reshape(1, -1))
    else:
        Y_obs = check_array(Y_obs)
        v_pc = bel.Y_pre_processing.transform(Y_obs)[
            :, : Setup.HyperParameters.n_pc_target
        ]

    nc = bel.Y_pre_processing["pca"].n_components_
    dummy = np.zeros((1, nc))
    dummy[:, : v_pc.shape[1]] = v_pc
    v_pred = bel.Y_pre_processing.inverse_transform(dummy)

    h_to_plot = np.copy(Y_obs.reshape(1, shape[1], shape[2]))

    whpa_plot(whpa=h_to_plot, color="red", alpha=1, lw=2)

    whpa_plot(
        whpa=v_pred.reshape(1, shape[1], shape[2]),
        color="blue",
        alpha=1,
        lw=2,
        labelsize=11,
        xlabel="X(m)",
        ylabel="Y(m)",
        x_lim=Setup.Focus.x_range,
        y_lim=Setup.Focus.y_range,
    )

    # Add title inside the box
    an = ["B"]

    legend_a = _proxy_annotate(annotation=an, loc=2, fz=14)

    _proxy_legend(
        legend1=legend_a,
        colors=["red", "blue"],
        labels=["Physical", "Back transformed"],
        marker=["-", "-"],
    )

    if fig_dir is not None:
        bel4ed.utils.dirmaker(fig_dir)
        plt.savefig(
            jp(fig_dir, f"h_pca_inverse_transform.png"), dpi=300, transparent=False
        )
        plt.close()

    if show:
        plt.show()
        plt.close()


def plot_results(
    bel,
    d: bool = True,
    h: bool = True,
    X=None,
    X_obs=None,
    Y=None,
    Y_obs=None,
    root: str = None,
    base_dir: str = None,
    folder: str = None,
    annotation: list = None,
):
    """
    Plots forecasts results in the 'uq' folder
    :param base_dir:
    :param Y_obs:
    :param Y:
    :param X_obs:
    :param X:
    :param bel:
    :param annotation: List of annotations
    :param h: Boolean to plot target or not
    :param d: Boolean to plot predictor or not
    :param root: str: Forward ID
    :param folder: str: Well combination. '123456', '1'...
    :return:
    """
    # Directory
    md = jp(base_dir, root, folder)
    # Wells
    wells = Setup.Wells
    wells_id = list(wells.wells_data.keys())
    cols = [wells.wells_data[w]["color"] for w in wells_id if "pumping" not in w]

    if d:
        # Curves - d
        # Plot curves
        sdir = jp(md, "data")

        X = check_array(X)
        # X_obs = check_array(X_obs)
        X_obs = X_obs.to_numpy().reshape(1, -1)

        tc = X.reshape((-1,) + bel.X_shape)
        tcp = X_obs.reshape((-1,) + bel.X_shape)
        tc = np.concatenate((tc, tcp), axis=0)

        # Plot parameters for predictor
        xlabel = "Observation index number"
        ylabel = "Concentration ($g/m^{3})$"
        factor = 1000
        labelsize = 11

        curves(
            cols=cols,
            tc=tc,
            sdir=sdir,
            xlabel=xlabel,
            ylabel=ylabel,
            factor=factor,
            labelsize=labelsize,
            highlight=[len(tc) - 1],
        )

        # curves(
        #     cols=cols,
        #     tc=tc,
        #     sdir=sdir,
        #     xlabel=xlabel,
        #     ylabel=ylabel,
        #     factor=factor,
        #     labelsize=labelsize,
        #     highlight=[len(tc) - 1],
        #     ghost=True,
        #     title="curves_ghost",
        # )

        # curves_i(
        #     cols=cols,
        #     tc=tc,
        #     xlabel=xlabel,
        #     ylabel=ylabel,
        #     factor=factor,
        #     labelsize=labelsize,
        #     sdir=sdir,
        #     highlight=[len(tc) - 1],
        # )

    if h:
        # WHP - h test + training
        fig_dir = jp(base_dir, root)
        ff = jp(fig_dir, f"{root}.png")  # figure name
        Y, Y_obs = check_array(Y), Y_obs.to_numpy().reshape(-1, 1)
        h_test = Y_obs.reshape((bel.Y_shape[1], bel.Y_shape[2]))
        h_training = Y.reshape((-1,) + (bel.Y_shape[1], bel.Y_shape[2]))
        # Plots target training + prediction
        # whpa_plot(whpa=h_training, color="blue", alpha=0.5, load_bstack=True)
        # whpa_plot(
        #     whpa=h_test,
        #     color="r",
        #     lw=2,
        #     alpha=0.8,
        #     xlabel="X(m)",
        #     ylabel="Y(m)",
        #     labelsize=11,
        # )
        # colors = ["blue", "red"]
        # labels = ["Training", "Test"]
        # legend = _proxy_annotate(annotation=["C"], loc=2, fz=14)
        # _proxy_legend(legend1=legend, colors=colors, labels=labels, fig_file=ff)
        # plt.close()

        # WHPs
        ff = jp(md, "uq", f"{root}_cca_{bel.cca.n_components}.png")
        forecast_posterior = bel.random_sample(n_posts=Setup.HyperParameters.n_posts)
        forecast_posterior = bel.inverse_transform(forecast_posterior)
        forecast_posterior = forecast_posterior.reshape(
            (-1,) + (bel.Y_shape[1], bel.Y_shape[2])
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
            load_bstack=True,
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
            x_lim=Setup.Focus.x_range,
            xlabel="X(m)",
            ylabel="Y(m)",
            labelsize=11,
            show=False,
        )

        # Other tricky operation to add annotation
        legend_an = _proxy_annotate(annotation=annotation, loc=2, fz=14)

        # Tricky operation to add a second legend:
        _proxy_legend(
            legend1=well_legend,
            extra=[legend_an],
            colors=colors,
            loc=3,
            labels=labels,
            fig_file=ff,
        )


def plot_K_field(
    root: str = None,
    base_dir: str = None,
    k_dir: str = None,
    wells=None,
    annotation: str = None,
    deprecated: bool = True,
    show: bool = False,
):
    if wells is None:
        wells = Setup.Wells

    if k_dir is None:
        k_dir = Setup.Directories.hydro_res_dir

    matrix = np.load(jp(k_dir, root, "hk0.npy"))
    grid_dim = Setup.GridDimensions
    extent = (grid_dim.xo, grid_dim.x_lim, grid_dim.yo, grid_dim.y_lim)

    hkf = jp(base_dir, root, "k_field.png")

    if deprecated:
        # HK field
        plt.figure()
        ax = plt.gca()
        im = ax.imshow(np.log10(matrix), cmap="coolwarm", extent=extent)
        plt.xlabel("X(m)", fontsize=11)
        plt.ylabel("Y(m)", fontsize=11)
        plot_wells(wells, markersize=3.5)

        well_legend = plt.legend(fontsize=11, loc=3, framealpha=0.6)
        legend_a = _proxy_annotate(annotation=[annotation], loc=2, fz=14)
        plt.gca().add_artist(legend_a)
        plt.gca().add_artist(well_legend)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.ax.set_title("$Log_{10} m/d$")
        plt.savefig(hkf, bbox_inches="tight", dpi=300, transparent=False)
        if show:
            plt.show()
        plt.close()


def mode_histo(
    an_i: int,
    wm: np.array,
    combi: list,
    colors: list = None,
    title: str = None,
    fig_name: str = "average",
    directory: str = None,
):
    """

    :param directory:
    :param title:
    :param combi:
    :param colors:
    :param an_i: Figure annotation
    :param wm: Arrays of metric
    :param fig_name:
    :return:
    """
    if directory is None:
        directory = Setup.Directories.forecasts_dir
    alphabet = string.ascii_uppercase
    # wid = list(map(str, Setup.Wells.combination))  # Wel identifiers (n)
    wid = combi

    pipeline = Pipeline(
        [
            ("s_scaler", StandardScaler()),
        ]
    )
    wm = pipeline.fit_transform(wm)

    # modes = []  # Get MHD corresponding to each well's mode
    # for i, m in enumerate(wm):  # For each well, look up its MHD distribution
    #     count, values = np.histogram(m, bins="fd")
    #     # (Freedman Diaconis Estimator)
    #     # Robust (resilient to outliers) estimator that takes into account data variability and data size.
    #     # https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
    #     idm = np.argmax(count)
    #     mode = values[idm]
    #     modes.append(mode)
    #
    # modes = np.array(modes)  # Scale modes
    # modes -= np.mean(modes)

    # Bar plot
    # plt.bar(np.arange(1, len(wid) + 1), -modes, color=colors)
    # # plt.title("Amount of information of each well")
    # plt.title(f"{fig_name}")
    # plt.xlabel("Well ID")
    # plt.ylabel("Opposite deviation from mode's mean")
    # plt.grid(color="#95a5a6", linestyle="-", linewidth=0.5, axis="y", alpha=0.7)
    #
    # legend_a = _proxy_annotate(annotation=[alphabet[an_i + 1]], loc=2, fz=14)
    # plt.gca().add_artist(legend_a)
    #
    # plt.savefig(
    #     os.path.join(directory, f"{fig_name}_well_mode.png"),
    #     dpi=300,
    #     transparent=False,
    # )
    # plt.close()

    # Plot BOX
    lol = np.array([np.median(w) for w in wm])
    columns = ["".join([str(wi) for wi in w]) for w in wid]

    # Let's sort in increasing order
    wm_idx = np.array([x for _, x in sorted(zip(lol, np.arange(len(combi))))])
    wm = wm[wm_idx]
    columns = [x for _, x in sorted(zip(lol, columns))]
    well_cols = ["b", "g", "r", "c", "m", "y"]
    well_cols = [x for _, x in sorted(zip(lol, well_cols))]
    lol = sorted(lol)

    # columns = ["1", "2", "3", "4", "5", "6"]
    norm = matplotlib.colors.Normalize(vmin=min(lol), vmax=max(lol))
    cmap = matplotlib.cm.get_cmap("coolwarm")
    wmd = pd.DataFrame(columns=columns, data=wm.T)
    palette = {columns[i]: cmap(norm(lol[i])) for i in range(len(columns))}
    fig, ax1 = plt.subplots()
    sns.boxplot(data=wmd, palette=palette, order=columns, linewidth=1, ax=ax1)
    [line.set_color("white") for line in ax1.get_lines()[4::6]]
    plt.ylim([-4, 4])
    plt.xlabel("Well ID")
    if len(combi) > 12:
        rotation = 70
    else:
        rotation = 0
    plt.xticks(rotation=rotation, fontsize=11, weight="bold")
    plt.ylabel("Metric value")
    if title is None:
        title = "Box-plot of the metric values for each data source"
    plt.title(title)
    plt.grid(color="saddlebrown", linestyle="--", linewidth=0.7, axis="y", alpha=0.5)

    try:
        an_i = int(directory.split("split")[-1])
    except ValueError:
        pass
    legend_a = _proxy_annotate(annotation=[alphabet[an_i]], loc=2, fz=14)
    plt.gca().add_artist(legend_a)

    plt.subplots_adjust(right=0.9)
    # Colorbar
    # rect = [left, bottom, width, height
    axcb = plt.axes([0.912, 0.11, 0.017, 0.771])
    cb1 = matplotlib.colorbar.ColorbarBase(
        axcb,
        cmap=cmap,
        norm=norm,
        orientation="vertical",
    )
    cb1.ax.set_title("Median", fontsize=8)
    cb1.ax.tick_params(labelsize=7)
    # for xtick, color in zip(ax1.get_xticklabels(), well_cols):
    #     xtick.set_color(color)

    # Insert well plot
    from mpl_toolkits.axes_grid.inset_locator import inset_axes

    inset_axes = inset_axes(
        ax1,
        width="25%",  # width = 30% of parent_bbox
        height="25%",  # height : 1 inch
        loc=3,
    )
    plot_wells(wells=Setup.Wells, annotate=True)
    plt.xticks([])
    plt.yticks([])

    # legend_b = _proxy_annotate(annotation=[f"Fold {an_i + 1}"], loc=1, fz=14)
    # plt.gca().add_artist(legend_b)

    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(
        os.path.join(directory, f"{fig_name}_well_box.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=False,
    )
    plt.close()

    # Plot histogram
    # for i, m in enumerate(wm):
    #     # sns.kdeplot(m, color=f"{colors[i]}", shade=True, linewidth=2)
    #     sns.kdeplot(m, color="b", shade=True, linewidth=2)
    # # plt.title("Summed metric distribution for each well")
    # plt.title(f"{fig_name}")
    # plt.xlabel("Summed metric")
    # plt.ylabel("KDE")
    # legend_1 = plt.legend(wid, loc=2)
    # plt.gca().add_artist(legend_1)
    # plt.grid(alpha=0.2)
    #
    # legend_a = _proxy_annotate(annotation=[alphabet[an_i]], loc=2, fz=14)
    # plt.gca().add_artist(legend_a)
    #
    # plt.savefig(
    #     os.path.join(directory, f"{fig_name}_hist.png"),
    #     dpi=300,
    #     transparent=False,
    # )
    # plt.close()


def curves(
    cols: list,
    tc: np.array,
    highlight: list = None,
    ghost: bool = False,
    sdir: str = None,
    labelsize: float = 12,
    factor: float = 1,
    conc: bool = 0,
    xlabel: str = None,
    ylabel: str = None,
    title: str = "curves",
    show: bool = False,
):
    """
    Shows every breakthrough curve stacked on a plot.
    :param cols: List of colors
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

    if not conc:
        n_sim, n_wells, nts = tc.shape
        for i in range(n_sim):
            for t in range(n_wells):
                if i in highlight:
                    plt.plot(tc[i][t] * factor, color=cols[t], linewidth=2, alpha=1)
                elif not ghost:
                    plt.plot(tc[i][t] * factor, color=cols[t], linewidth=0.2, alpha=0.5)
    else:
        plt.plot(np.concatenate(tc[0]) * factor, color=cols[0], linewidth=2, alpha=1)

    plt.grid(linewidth=0.3, alpha=0.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tick_params(labelsize=labelsize)
    if sdir:
        bel4ed.utils.dirmaker(sdir)
        plt.savefig(jp(sdir, f"{title}.png"), dpi=300, transparent=False)
        plt.close()
    if show:
        plt.show()
        plt.close()


def curves_i(
    cols: list,
    tc: np.array,
    highlight: list = None,
    labelsize: float = 12,
    factor: float = 1,
    xlabel: str = None,
    ylabel: str = None,
    sdir: str = None,
    show: bool = False,
):
    """
    Shows every breakthrough individually for each observation point.
    Will produce n_well figures of n_sim curves each.
    :param cols: List of colors
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
                plt.plot(tc[i][t] * factor, color=cols[t], linewidth=0.2, alpha=0.5)
        colors = [cols[t], "k"]
        plt.grid(linewidth=0.3, alpha=0.4)
        plt.tick_params(labelsize=labelsize)
        # plt.title(f'Well {t + 1}')

        alphabet = string.ascii_uppercase
        legend_a = _proxy_annotate([f"{alphabet[t]}. Well {t + 1}"], fz=12, loc=2)

        labels = ["Training", "Test"]
        _proxy_legend(legend1=legend_a, colors=colors, labels=labels, loc=1)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if sdir:
            bel4ed.utils.dirmaker(sdir)
            plt.savefig(jp(sdir, f"{title}_{t + 1}.png"), dpi=300, transparent=False)
            plt.close()
        if show:
            plt.show()
            plt.close()


def plot_wells(
    wells: Setup.Wells,
    well_ids: list = None,
    markersize: float = 4.0,
    annotate: bool = False,
):
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
            if annotate:
                plt.annotate(
                    label,
                    fontsize=8,
                    xy=(wbd[i]["coordinates"][0], wbd[i]["coordinates"][1]),
                    xytext=(-5, 5),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.5),
                )
                # arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0'))
        s += 1


def plot_head_field(root: str = None, base_dir: str = None, annotation: str = None):
    matrix = np.load(jp(Setup.Directories.hydro_res_dir, root, "whpa_heads.npy"))
    grid_dim = Setup.GridDimensions
    extent = (grid_dim.xo, grid_dim.x_lim, grid_dim.yo, grid_dim.y_lim)

    hkf = jp(base_dir, root, "heads_field.png")

    # HK field
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(matrix[0], cmap="Blues_r", extent=extent)
    plt.xlabel("X(m)", fontsize=11)
    plt.ylabel("Y(m)", fontsize=11)
    if annotation:
        _proxy_annotate(annotation=[annotation], loc=2, fz=14)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.set_title("Head (m)")



    plt.savefig(hkf, bbox_inches="tight", dpi=300, transparent=False)
    plt.close()


def plot_pc_ba(
    bel,
    X_obs,
    Y_obs,
    base_dir: str = None,
    root: str = None,
    w: str = None,
    data: bool = False,
    target: bool = False,
):
    """
    Comparison between original variables and the same variables back-transformed with n PCA components.
    :param root:
    :param data:
    :param target:
    :return:
    """

    if isinstance(root, (list, tuple)):
        if len(root) > 1:
            logger.error("Input error")
            return
        else:
            root = root[0]

    subdir = os.path.join(base_dir, root)

    if data:
        # Plot parameters for predictor
        xlabel = "Observation index number"
        ylabel = "Concentration ($g/m^{3})$"
        factor = 1000
        labelsize = 11
        d_pca_inverse_plot(
            bel,
            X_obs=X_obs,
            root=root,
            xlabel=xlabel,
            ylabel=ylabel,
            labelsize=labelsize,
            factor=factor,
            fig_dir=os.path.join(subdir, w, "pca"),
        )
    if target:
        h_pca_inverse_plot(bel, Y_obs=Y_obs, fig_dir=os.path.join(subdir, w, "pca"))


def plot_whpa(bel, Y, Y_obs, base_dir, root):
    """
    Loads target pickle and plots all training WHPA
    :return:
    """

    h_training = Y.reshape(bel.Y_shape)

    whpa_plot(
        whpa=h_training, highlight=True, halpha=0.5, lw=0.1, color="darkblue", alpha=0.5
    )

    if root is not None:
        h_pred = Y_obs.reshape(bel.Y_shape)
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
        legend = _proxy_annotate(annotation=["C"], loc=2, fz=14)
        _proxy_legend(
            legend1=legend,
            colors=["darkblue", "darkred"],
            labels=labels,
            fig_file=os.path.join(base_dir, root, "whpa_training.png"),
        )


def cca_vision(
    base_dir: str = None, Y_obs: np.array = None, root: str = None, folders: list = None
):
    """
    Loads CCA pickles and plots components for all folders
    :param root:
    :param folders:
    :return:
    """

    if isinstance(root, (list, tuple)):
        if len(root) > 1:
            logger.error("Input error")
            return
        else:
            root = root[0]

    subdir = os.path.join(base_dir, root)

    if folders is None:
        listme = os.listdir(subdir)
        folders = list(filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))
    else:
        if not isinstance(folders, (list, tuple)):
            folders = [folders]
        else:
            pass

    for f in folders:
        res_dir = os.path.join(subdir, f, "obj")

        # Load objects
        bel = reload_trained_model(base_dir=base_dir, root=root, well=f)

        # CCA coefficient plot
        cca_coefficient = np.corrcoef(bel.X_c.T, bel.Y_c.T).diagonal(
            offset=bel.cca.n_components
        )  # Gets correlation coefficient
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
        cb.ax.set_title(r"$\it{" + "r" + "}$")
        plt.grid(alpha=0.4, linewidth=0.5, zorder=0)
        plt.xticks(
            np.arange(len(cca_coefficient)), np.arange(1, len(cca_coefficient) + 1)
        )
        plt.tick_params(labelsize=5)
        plt.yticks([])
        # plt.title('Decrease of CCA correlation coefficient with component number')
        plt.ylabel("Correlation coefficient")
        plt.xlabel("Component number")

        # Add annotation
        legend = _proxy_annotate(annotation=["D"], fz=14, loc=1)
        plt.gca().add_artist(legend)

        plt.savefig(
            os.path.join(os.path.dirname(res_dir), "cca", "coefs.png"),
            bbox_inches="tight",
            dpi=300,
            transparent=False,
        )
        plt.close()

        # KDE plots which consume a lot of time.
        _kde_cca(bel, Y_obs=Y_obs, sdir=os.path.join(subdir, f, "cca"))


def pca_vision(
    bel,
    X_obs=None,
    Y_obs=None,
    root: str or Root = None,
    base_dir: str = None,
    w: str = None,
    d: bool = True,
    h: bool = False,
    scores: bool = True,
    exvar: bool = True,
    before_after=True,
    labels: bool = False,
):
    """
    Loads PCA pickles and plot scores for all folders
    :param w:
    :param labels:
    :param root: str:
    :param d: bool:
    :param h: bool:
    :param scores: bool:
    :param exvar: bool:
    :return:
    """

    if isinstance(root, (list, tuple)):
        if len(root) > 1:
            logger.error("Input error")
            return
        else:
            root = root[0]

    subdir = jp(base_dir, root, w, "pca")

    if d:
        fig_file = os.path.join(subdir, "d_scores.png")
        if scores:
            pca_scores(
                training=bel.X_pc,
                prediction=bel.X_obs_pc,
                n_comp=Setup.HyperParameters.n_pc_predictor,
                annotation=["C"],
                labels=labels,
                fig_file=fig_file,
            )
        # Explained variance plots
        if exvar:
            fig_file = os.path.join(subdir, "d_exvar.png")
            explained_variance(
                bel,
                n_comp=Setup.HyperParameters.n_pc_predictor,
                thr=0.8,
                annotation=["E"],
                fig_file=fig_file,
            )
        if before_after:
            plot_pc_ba(
                bel,
                X_obs=X_obs,
                Y_obs=None,
                base_dir=base_dir,
                root=root,
                w=w,
                data=True,
                target=False,
            )
    if h:
        # Transform and split
        h_pc_training = bel.Y_pc
        Y_obs = Y_obs.to_numpy().reshape(1, -1)
        h_pc_prediction = bel.Y_pre_processing.transform(Y_obs)
        # Plot
        fig_file = os.path.join(subdir, "h_pca_scores.png")
        if scores:
            pca_scores(
                training=h_pc_training,
                prediction=h_pc_prediction,
                n_comp=Setup.HyperParameters.n_pc_target,
                annotation=["D"],
                labels=labels,
                fig_file=fig_file,
            )
        # Explained variance plots
        if exvar:
            fig_file = os.path.join(subdir, "h_pca_exvar.png")
            explained_variance(
                bel,
                n_comp=Setup.HyperParameters.n_pc_target,
                thr=0.8,
                annotation=["F"],
                fig_file=fig_file,
            )
        if before_after:
            plot_pc_ba(
                bel,
                X_obs=None,
                Y_obs=Y_obs,
                base_dir=base_dir,
                root=root,
                w=w,
                data=False,
                target=True,
            )


def check_root(xlim: list, ylim: list, root: list):
    """
    Plots raw data of folder 'root'
    :param xlim:
    :param ylim:
    :param root:
    :return:
    """
    bkt, whpa, _ = data_loader(roots=root)
    whpa = whpa.squeeze()
    # self.curves_i(bkt, show=True)  # This function will not work

    plt.plot(whpa[:, 0], whpa[:, 1], "wo")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def d_pca_inverse_plot(
    bel,
    root,
    X_obs,
    factor: float = 1.0,
    xlabel: str = None,
    ylabel: str = None,
    labelsize: float = 11.0,
    fig_dir: str = None,
    show: bool = False,
):
    """
    Plot used to compare the reproduction of the original physical space after PCA transformation.
    :param xlabel:
    :param ylabel:
    :param labelsize:
    :param factor:
    :param fig_dir: str:
    :param show: bool:
    :return:
    """

    shape = bel.X_shape
    v_pc = bel.X_obs_pc

    nc = bel.X_pre_processing["pca"].n_components_
    dummy = np.zeros((1, nc))
    dummy[:, : v_pc.shape[1]] = v_pc

    v_pred = bel.X_pre_processing.inverse_transform(dummy).reshape((-1,) + shape)
    to_plot = np.copy(X_obs).reshape((-1,) + shape)

    cols = ["r" for _ in range(shape[1])]
    highlights = [i for i in range(shape[1])]
    curves(cols=cols, tc=to_plot, factor=factor, highlight=highlights, conc=True)

    cols = ["b" for _ in range(shape[1])]
    curves(cols=cols, tc=v_pred, factor=factor, highlight=highlights, conc=True)

    # Add title inside the box
    an = ["A"]
    legend_a = _proxy_annotate(annotation=an, loc=2, fz=14)
    _proxy_legend(
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
        bel4ed.utils.dirmaker(fig_dir)
        plt.savefig(jp(fig_dir, f"{root}_d.png"), dpi=300, transparent=False)
        plt.close()
    if show:
        plt.show()
        plt.close()


@deprecated()
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
            bbox=dict(
                facecolor="white", edgecolor="black", boxstyle="round,pad=.5", alpha=0.7
            ),
        )
        # plt.annotate(i, (epxy[i, 0] + 4, epxy[i, 1] + 4), fontsize=14, weight='bold', color='r')

    plt.grid(alpha=0.5)
    plt.xlim([870, 1080])
    plt.ylim([415, 600])
    plt.xlabel("X(m)")
    plt.ylabel("Y(m)")
    plt.tick_params(labelsize=11)

    legend = _proxy_annotate(annotation=["A"], loc=2, fz=14)
    plt.gca().add_artist(legend)

    plt.savefig(
        jp(md.forecasts_dir, "base", "roots_whpa", f"{root}_ep.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=False,
    )
    plt.show()


def order_vertices(vertices: np.array) -> np.array:
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
        )
    )

    # Sort vertices according to angle
    so = sorted(
        vertices,
        key=lambda coord: (
            math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))
        ),
    )

    return np.array(so)


class ModelVTK:
    """
    Loads flow/transport models and export the VTK objects.
    """

    def __init__(self, base: Setup = None, folder: str = None):
        self.base = base
        md = self.base.Directories()
        self.rn = folder
        self.bdir = md.main_dir
        self.results_dir = jp(md.hydro_res_dir, self.rn)
        self.vtk_dir = jp(self.results_dir, "vtk")
        bel4ed.utils.dirmaker(self.vtk_dir)

        # Load flow model
        try:
            m_load = jp(self.results_dir, "whpa.nam")
            self.flow_model = load_flow_model(m_load, model_ws=self.results_dir)
            delr = self.flow_model.modelgrid.delr  # thicknesses along rows
            delc = self.flow_model.modelgrid.delc  # thicknesses along column
            # xyz_vertices = self.flow_model.modelgrid.xyzvertices
            # blocks2d = mops.blocks_from_rc(delc, delr)
            self.blocks = blocks_from_rc_3d(delc, delr)
            # blocks3d = self.blocks.reshape(-1, 3)
        except Exception as e:
            logger.error(e)

        try:
            # Transport model
            mt_load = jp(self.results_dir, "whpa.mtnam")
            self.transport_model = load_transport_model(
                mt_load, self.flow_model, model_ws=self.results_dir
            )
            ucn_files = [
                jp(self.results_dir, f"MT3D00{i}.UCN") for i in Setup.Wells.combination
            ]  # Files containing concentration
            ucn_obj = [flopy.utils.UcnFile(uf) for uf in ucn_files]  # Load them
            self.times = [uo.get_times() for uo in ucn_obj]  # Get time steps
            self.concs = np.array([uo.get_alldata() for uo in ucn_obj])  # Get all data
        except Exception as e:
            logger.error(e)

    def flow_vtk(self):
        """
        Export flow model packages and computed heads to vtk format

        """
        dir_fv = jp(self.results_dir, "vtk", "flow")
        bel4ed.utils.dirmaker(dir_fv)
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
        bel4ed.utils.dirmaker(dir_tv)
        self.transport_model.export(dir_tv, fmt="vtk")

    # Export UCN to vtk

    def stacked_conc_vtk(self):
        """Stack component concentrations for each time step and save vtk"""
        conc_dir = jp(self.results_dir, "vtk", "transport", "concentration")
        bel4ed.utils.dirmaker(conc_dir)
        # First replace 1e+30 value (inactive cells) by 0.
        conc0 = np.abs(np.where(self.concs == 1e30, 0, self.concs))
        # Cells configuration for 3D blocks
        cells = [
            ("quad", np.array([list(np.arange(i * 4, i * 4 + 4))]))
            for i in range(len(self.blocks))
        ]
        for j in range(self.concs.shape[1]):
            # Stack the concentration of each component at each time step to visualize them in one plot.
            array = np.zeros(self.blocks.shape[0])
            dic_conc = {}
            for i in range(1, 7):
                # fliplr is necessary as the reshape modifies the array structure
                cflip = np.fliplr(conc0[i - 1, j]).reshape(-1)
                dic_conc[f"conc_wel_{i}"] = cflip
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
            [
                points.InsertPoint(e * 4 + es, bb) for es, bb in enumerate(sb)
            ]  # Insert points by giving first their index e*4+es
            ugrid.InsertNextCell(
                vtk.VTK_PIXEL, 4, list(range(e * 4, e * 4 + 4))
            )  # Insert cell in UGrid

        ugrid.SetPoints(points)  # Set points

        for i in range(1, 7):  # For eaxh injecting well
            conc_dir = jp(self.results_dir, "vtk", "transport", f"{i}_UCN")
            bel4ed.utils.dirmaker(conc_dir)
            # Initiate array and give it a name
            concArray = vtk.vtkDoubleArray()
            concArray.SetName(f"conc{i}")
            for j in range(self.concs.shape[1]):  # For each time step
                # Set array
                array = np.fliplr(conc0[i - 1, j]).reshape(-1)
                [concArray.InsertNextValue(s) for s in array]
                ugrid.GetCellData().AddArray(
                    concArray
                )  # Add array to unstructured grid

                # Save grid
                writer = vtk.vtkXMLUnstructuredGridWriter()
                writer.SetInputData(ugrid)
                writer.SetFileName(jp(conc_dir, f"{j}_conc.vtu"))
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
        bel4ed.utils.dirmaker(back_dir)

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
                )
            )
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
                    short = np.vstack(
                        (
                            points_x[p, : i + 1],
                            points_y[p, : i + 1],
                            np.abs(points_z[p, : i + 1] * 0),
                        )
                    ).T
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
                    [speed.InsertNextValue(speed_array[k][p]) for k in range(i + 1)]
                    polyData.GetPointData().AddArray(speed)

                    # Add the lines to the dataset
                    polyData.SetLines(cells)

                    # Export
                    writer = vtk.vtkXMLPolyDataWriter()
                    writer.SetInputData(polyData)
                    writer.SetFileName(jp(back_dir, "path{}_t{}.vtp".format(p, i)))
                    writer.Write()
            else:
                for p in range(n_particles):
                    xs = points_x[p, i]
                    ys = points_y[p, i]
                    # Replace elevation by 0 to project them in the surface
                    zs = points_z[p, i] * 0
                    xyz_particles_t_i = np.vstack((xs, ys, zs)).T.reshape(-1, 3)

                    # Create points
                    points = vtk.vtkPoints()
                    ids = [points.InsertNextPoint(c) for c in xyz_particles_t_i]

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
                    writer.SetFileName(jp(back_dir, "path{}_t{}.vtp".format(p, i)))
                    writer.Write()

    # %% Export wells objects as vtk

    def wells_vtk(self):
        """Exports wells coordinates to VTK"""

        wbd = self.base.Wells().wells_data

        wels = np.array([wbd[o]["coordinates"] for o in wbd])
        wels = np.insert(
            wels, 2, np.zeros(len(wels)), axis=1
        )  # Insert zero array for Z

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
