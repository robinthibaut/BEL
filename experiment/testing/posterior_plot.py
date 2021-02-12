import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import ma
from matplotlib import pyplot as pp
from matplotlib import gridspec

from experiment.calculation import kdeplot


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

# https://stackoverflow.com/questions/35920885/how-to-overlay-a-seaborn-jointplot-with-a-marginal-distribution-histogram-fr
# You can plot directly onto the JointGrid.ax_marg_x and JointGrid.ax_marg_y attributes,
# which are the underlying matplotlib axes.


# https://peterroelants.github.io/posts/multivariate-normal-primer/

# Generate a random correlated bivariate dataset
rs = np.random.RandomState(5)
mean = [0, 0]
cov = [(1, .96), (.96, 1)]
x1, x2 = rs.multivariate_normal(mean, cov, 200).T
d = pd.Series(x1, name="$X_1$")
h = pd.Series(x2, name="$X_2$")

d_cca_prediction = [0]
h_cca_prediction = [0]
comp_n = 0
sample_n = 0

cmap = sns.color_palette("Blues", as_cmap=True)

# Plot h posterior given d
density, support = kdeplot.kde_params(x=d, y=h)
xx, yy = support

marginal_eval = kdeplot.KDE()

kde_x, sup_x = marginal_eval(d)
kde_y, sup_y = marginal_eval(h)

xmin, xmax = min(sup_x), max(sup_x)
ymin, ymax = min(sup_y), max(sup_y)

# # %%
# # Choose beautiful color map
# # cube_helix very nice for dark mode
# # light = 0.95 is beautiful for reverse = True
# # cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=False)
# # Seaborn 'joinplot' between d & h training CCA scores
# g = sns.jointplot(d, h,
#                   cmap=cmap, n_levels=80, shade=True,
#                   kind='kde')
# g.plot_joint(plt.scatter, c='w', marker='o', s=2, alpha=.7)
# g.ax_marg_x.arrow(d_cca_prediction[comp_n], 0, 0, .1, color='r', head_width=0, head_length=0, lw=2)
# g.ax_marg_y.arrow(0, h_cca_prediction[comp_n], .1, 0, color='r', head_width=0, head_length=0, lw=2)
# # Plot prediction (d, h) in canonical space
# plt.plot(d_cca_prediction[comp_n], h_cca_prediction[comp_n],
#          'ro', markersize=4.5, markeredgecolor='k', alpha=1,
#          label=f'{sample_n}')
# # plt.grid('w', linewidth=.3, alpha=.4)
# # plt.tick_params(labelsize=8)
# plt.xlabel('$d^{c}$', fontsize=14)
# plt.ylabel('$h^{c}$', fontsize=14)
# plt.subplots_adjust(top=0.9)
# plt.tick_params(labelsize=14)
#
# plt.show()
#
# # %%
# # Define grid for subplots
# # gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4])
# gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])
#
# # Create density plot
# fig = pp.figure()
# ax = pp.subplot(gs[1, 0])
# cax = ax.imshow(density,
#                 origin='lower',
#                 cmap=cmap,
#                 extent=[min(xx), max(xx), min(yy), max(yy)])
# ax.set_ylim((xmin, xmax))
# ax.set_ylim((ymin, ymax))
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.axvline(x=d_cca_prediction[0], color='r')
# plt.scatter(d, h, c='k', marker='o', s=2, alpha=.7)
# plt.plot(d_cca_prediction[comp_n], h_cca_prediction[comp_n],
#          'ro', markersize=4.5, markeredgecolor='k', alpha=1,
#          label=f'{sample_n}')
#
# # Create Y-marginal (right)
# axr = pp.subplot(gs[1, 1],
#                  sharey=ax,
#                  xticks=[], yticks=[],
#                  frameon=False,
#                  xlim=(0, 1.4 * kde_y.max()),
#                  ylim=(ymin, ymax))
# axr.plot(kde_y, sup_y, color='black')
# axr.fill_betweenx(sup_y, 0, kde_y, alpha=.5, color='#5673E0')
#
# # Create X-marginal (top)
# axt = pp.subplot(gs[0, 0],
#                  sharex=ax, frameon=False,
#                  xticks=[], yticks=[],
#                  xlim=(xmin, xmax),
#                  ylim=(0, 1.4 * kde_x.max()),
#                  aspect='auto')
# axt.plot(sup_x, kde_x, color='black')
# axt.fill_between(sup_x, 0, kde_x, alpha=.5, color='#5673E0')
#
# # Bring the marginals closer to the contour plot
# fig.tight_layout(pad=.5)
#
# # plt.xlabel('$d^{c}$', fontsize=14)
# # plt.ylabel('$h^{c}$', fontsize=14)
# # plt.subplots_adjust(top=0.9)
# # plt.tick_params(labelsize=14)
# plt.show()

# %%
height = 6
ratio = 5
space = .2,
dropna = False
xlim = None
ylim = None
size = None
marginal_ticks = False
hue = None
palette = None
hue_order = None
hue_norm = None

# Set up the subplot grid
f = plt.figure(figsize=(height, height))
gs = plt.GridSpec(ratio + 1, ratio + 1)

ax_joint = f.add_subplot(gs[1:, :-1])
ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)

fig = f
ax_joint = ax_joint
ax_marg_x = ax_marg_x
ax_marg_y = ax_marg_y

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

if xlim is not None:
    ax_joint.set_xlim(xlim)
if ylim is not None:
    ax_joint.set_ylim(ylim)

# Make the grid look nice
despine(f)
if not marginal_ticks:
    despine(ax=ax_marg_x, left=True)
    despine(ax=ax_marg_y, bottom=True)
for axes in [ax_marg_x, ax_marg_y]:
    for axis in [axes.xaxis, axes.yaxis]:
        axis.label.set_visible(False)
f.tight_layout()
space = 0  # Space between marginal plots and joint plot
f.subplots_adjust(hspace=space, wspace=space)

# Filled contour plot
z = ma.masked_where(density <= np.finfo(np.float16).eps, density)
cs = ax_joint.contourf(xx, yy, z, cmap='Greens', levels=69)
# Vertical line
ax_joint.axvline(x=d_cca_prediction[0], color='blue', linewidth=.5, alpha=.5)
# Horizontal line
ax_joint.axhline(y=h_cca_prediction[0], color='red', linewidth=.5, alpha=.5)
# Horizontal line
# ax_joint.axhline(y=h_cca_prediction[0], color='b', linewidth=.5)
# Scatter plot
ax_joint.scatter(d, h, c='k', marker='o', s=2, alpha=.7)
# Point
ax_joint.plot(d_cca_prediction[comp_n], h_cca_prediction[comp_n],
              'ro', markersize=4.5, markeredgecolor='k', alpha=1,
              label=f'{sample_n}')
# Marginal x plot
ax_marg_x.plot(sup_x, kde_x, color='black', linewidth=.5, alpha=0)
ax_marg_x.fill_between(sup_x, 0, kde_x, alpha=.1, color='darkblue')
# ax_marg_x.axvline(x=d_cca_prediction[0], ymax=0.25, color='blue', linewidth=.5, alpha=.5)
ax_marg_x.arrow(d_cca_prediction[0], 0, 0, .05, color='blue', head_width=.05, head_length=.05, lw=.5, alpha=.5)
# Marginal y plot
ax_marg_y.plot(kde_y, sup_y, color='black', linewidth=.5, alpha=0)
ax_marg_y.fill_betweenx(sup_y, 0, kde_y, alpha=.1, color='darkred')
# ax_marg_y.axhline(y=h_cca_prediction[0], xmax=0.25, color='red', linewidth=.5, alpha=.5)
ax_marg_y.arrow(d_cca_prediction[0], 0, .05, 0, color='red', head_width=.05, head_length=.05, lw=.5, alpha=.5)
# Conditional distribution
hp, sup = kdeplot.posterior_conditional(d, h, d_cca_prediction[0])
ax_marg_y.plot(hp, sup, 'r', alpha=0)
ax_marg_y.fill_betweenx(sup, 0, hp, alpha=.4, color='red')
# Labels
ax_joint.set_xlabel('$d^{c}$', fontsize=14)
ax_joint.set_ylabel('$h^{c}$', fontsize=14)
# plt.subplots_adjust(top=0.9)
plt.tick_params(labelsize=14)

plt.show()
