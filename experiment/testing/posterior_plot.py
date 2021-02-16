#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os

import numpy as np

import joblib
import matplotlib.pyplot as plt
from numpy import ma
from sklearn.preprocessing import PowerTransformer

import experiment._statistics as stats
from experiment._core import setup
from experiment._visualization import despine, my_alphabet, proxy_annotate, proxy_legend

comp_n = 0
sample_n = 0
base_dir = os.path.join(setup.directories.forecasts_dir, 'base')
res_dir = os.path.join(setup.directories.forecasts_dir, '818bf1676c424f76b83bd777ae588a1d/123456/obj/')

root = '818bf1676c424f76b83bd777ae588a1d'
f_names = list(map(lambda fn: os.path.join(res_dir, f'{fn}.pkl'), ['cca', 'd_pca']))
cca_operator, d_pco = list(map(joblib.load, f_names))

h_pco = joblib.load(os.path.join(base_dir, 'h_pca.pkl'))
h_pred = np.load(os.path.join(base_dir, 'roots_whpa', f'{root}.npy'))

# Inspect transformation between physical and PC space
dnc0 = d_pco.n_pc_cut
hnc0 = h_pco.n_pc_cut

# Cut desired number of PC components
d_pc_training, d_pc_prediction = d_pco.comp_refresh(dnc0)
h_pco.test_fit_transform(h_pred, test_root=[root])
h_pc_training, h_pc_prediction = h_pco.comp_refresh(hnc0)

# CCA plots
d_cca_training, h_cca_training = cca_operator.transform(d_pc_training,
                                                        h_pc_training)
d, h = d_cca_training.T, h_cca_training.T

d_obs = d_pc_prediction[sample_n]
h_obs = h_pc_prediction[sample_n]

# # Transform to CCA space and transpose
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

d = d[comp_n]
h = h[comp_n]
d_cca_prediction = d_cca_prediction[0]
h_cca_prediction = h_cca_prediction[0]
# Conditional:
hp, sup = stats.posterior_conditional(d, h, d_cca_prediction[0])

# load prediction object
lol = joblib.load(os.path.join(setup.directories.forecasts_dir, '818bf1676c424f76b83bd777ae588a1d/123456/obj/post.pkl'))
post_test = lol.random_sample(200).T
post_test_t = tfm1.transform(post_test.T).T
y_samp = post_test_t[0]

# Plot h posterior given d
density, support = stats.kde_params(x=d, y=h)
xx, yy = support

marginal_eval_x = stats.KDE()
marginal_eval_y = stats.KDE()

# support is cached
kde_x, sup_x = marginal_eval_x(d)
kde_y, sup_y = marginal_eval_y(h)
# use the same support as y
kde_y_samp, sup_samp = marginal_eval_y(y_samp)

xmin, xmax = min(sup_x), max(sup_x)
ymin, ymax = min(sup_y), max(sup_y)


# %%
def kde_cca(savefig: bool = False):
    height = 6
    ratio = 6
    space = 0

    xlim = None
    ylim = None
    marginal_ticks = False

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
    f.subplots_adjust(hspace=space, wspace=space)

    # Filled contour plot
    # Mask values under threshold
    z = ma.masked_where(density <= np.finfo(np.float16).eps, density)
    # Filled contour plot
    ax_joint.contourf(xx, yy, z,
                      cmap='Greens', levels=69)
    # Vertical line
    ax_joint.axvline(x=d_cca_prediction[comp_n],
                     color='red', linewidth=1, alpha=.5)
    # Horizontal line
    ax_joint.axhline(y=h_cca_prediction[comp_n],
                     color='deepskyblue', linewidth=1, alpha=.5)
    # Scatter plot
    ax_joint.scatter(d, h,
                     c='k', marker='o', s=2, alpha=.7)
    # Point
    ax_joint.plot(d_cca_prediction[comp_n], h_cca_prediction[comp_n],
                  'wo', markersize=5, markeredgecolor='k', alpha=1,
                  label=f'{sample_n}')
    # Marginal x plot
    #  - Line plot
    ax_marg_x.plot(sup_x, kde_x,
                   color='black', linewidth=.5, alpha=1)
    #  - Fill to axis
    ax_marg_x.fill_between(sup_x, 0, kde_x,
                           color='lightskyblue', alpha=1)
    #  - Notch indicating true value
    ax_marg_x.axvline(x=d_cca_prediction[comp_n],
                      ymax=0.25,
                      color='red', linewidth=1, alpha=.5,
                      label='$p(d^{c})$')
    # Marginal y plot
    #  - Line plot
    ax_marg_y.plot(kde_y, sup_y,
                   color='black', linewidth=.5, alpha=1)
    #  - Fill to axis
    ax_marg_y.fill_betweenx(sup_y, 0, kde_y,
                            alpha=.1, color='mistyrose')
    #  - Notch indicating true value
    ax_marg_y.axhline(y=h_cca_prediction[comp_n],
                      xmax=0.25,
                      color='deepskyblue', linewidth=1, alpha=.5,
                      label='$p(h^{c})')
    # Marginal y plot with BEL
    #  - Line plot
    ax_marg_y.plot(kde_y_samp, sup_samp,
                   color='black', linewidth=.5, alpha=1)
    #  - Fill to axis
    ax_marg_y.fill_betweenx(sup_samp, 0, kde_y_samp,
                            color='coral', alpha=.3,
                            label='$p(h^{c}|d^{c}_{*})$ (BEL)')
    # Conditional distribution
    #  - Line plot
    ax_marg_y.plot(hp, sup,
                   color='red', alpha=0)
    #  - Fill to axis
    ax_marg_y.fill_betweenx(sup, 0, hp,
                            color='salmon', alpha=.4,
                            label='$p(h^{c}|d^{c}_{*})$ (KDE)')
    # Labels
    ax_joint.set_xlabel('$d^{c}$', fontsize=14)
    ax_joint.set_ylabel('$h^{c}$', fontsize=14)
    # plt.subplots_adjust(top=0.9)
    plt.tick_params(labelsize=14)

    # Add custom artists
    subtitle = my_alphabet(comp_n)
    # Add title inside the box
    an = [f'{subtitle}. Pair {comp_n + 1} - R = {round(0.999, 3)}']
    legend_a = proxy_annotate(obj=ax_joint,
                              annotation=an,
                              loc=2,
                              fz=14)
    #
    proxy_legend(obj=ax_joint,
                 legend1=legend_a,
                 colors=['black', 'white'],
                 labels=['Training', 'Test'],
                 marker='o',
                 pec=['k', 'k'])

    if savefig:
        plt.savefig('plot3.png', bbox_inches='tight', dpi=300)


#


# plt.savefig('plot3.png', bbox_inches='tight', dpi=300)
kde_cca()
plt.show()


def posterior_distribution(savefig: bool = False):
    # prior
    plt.plot(sup_y, kde_y,
             color='black', linewidth=.5, alpha=1)
    plt.fill_between(sup_y, 0, kde_y,
                     color='mistyrose', alpha=1,
                     label='$p(h^{c})$')
    # posterior kde
    plt.plot(sup, hp,
             color='darkred', linewidth=.5, alpha=0)
    plt.fill_between(sup, 0, hp,
                     color='salmon', alpha=.5,
                     label='$p(h^{c}|d^{c}_{*})$ (KDE)')
    # posterior bel
    plt.plot(sup_samp, kde_y_samp,
             color='black', linewidth=.5, alpha=1)
    plt.fill_between(sup_samp, 0, kde_y_samp,
                     color='coral', alpha=.5,
                     label='$p(h^{c}|d^{c}_{*})$ (BEL)')

    # True prediction
    plt.axvline(x=h_cca_prediction[0],
                linewidth=3, alpha=.4, color='deepskyblue',
                label='True $h^{c}$')

    # Grid
    plt.grid(alpha=.2)

    # Tuning
    plt.ylabel('Density', fontsize=14)
    plt.xlabel('$h^{c}$', fontsize=14)
    plt.xlim([np.min(h_cca_training[comp_n]), np.max(h_cca_training[comp_n])])
    plt.tick_params(labelsize=14)

    plt.legend(loc=2)

    if savefig:
        plt.savefig('prior_post_h.png', bbox_inches='tight', dpi=300)
