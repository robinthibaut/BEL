#  Copyright (c) 2021. Robin Thibaut, Ghent University

import matplotlib.pyplot as plt
import numpy as np


def mat_cov_plot(mat, cmap='coolwarm', filename=None):
    mat[np.abs(mat) <= np.finfo(float).eps] = 0
    shape = mat.shape
    plt.matshow(mat, cmap=cmap, vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel(shape[1])
    # plt.ylabel(shape[0])
    if filename:
        plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight', transparent=True)
    cb = plt.colorbar()
    # cb.ax.set_title('Cov')
    if filename:
        plt.savefig(f'{filename}_cb.pdf', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def mat_mean_plot(mat, cmap='coolwarm', filename=None, vmin=None, vmax=None):
    mat[np.abs(mat) <= np.finfo(float).eps] = 0
    shape = mat.shape
    plt.matshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks([])
    plt.yticks([])
    if filename:
        plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight', transparent=True)
    cb = plt.colorbar(ticks=np.linspace(vmin, vmax, 12))
    cb.ax.tick_params(labelsize=30)
    if filename:
        plt.savefig(f'{filename}_cb.pdf', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


#
# mat_cov_plot(d_noise_covariance)
mat_cov_plot(g)
# mat_cov_plot(d_modeling_error)
# mat_cov_plot(d_modeling_covariance)

cmaps = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
         'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

c = 'coolwarm'

mat_cov_plot(h_cov_operator, cmap=c, filename='Chh')
mat_cov_plot(h_cov_operator @ g.T, cmap=c, filename='Chd')
mat_cov_plot(g @ h_cov_operator, cmap=c, filename='Cdh')
mat_cov_plot(g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance, cmap=c, filename='Cdd')
mat_cov_plot(np.linalg.pinv(g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance), cmap=c,
             filename='Cdd_inv')

mat_cov_plot((h_cov_operator @ g.T) \
             @ np.linalg.pinv(g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance) \
             @ g @ h_cov_operator, cmap=c, filename='Cmid')

mat_cov_plot(h_posterior_covariance, cmap=c, filename='Cpost')

mat_cov_plot(
    h_cov_operator @ g.T @ np.linalg.pinv(g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance),
    cmap=c, filename='mean_mid')

d_mean_e = d_cca_prediction.reshape(-1, 1) + d_modeling_mean_error - g @ h_mean
m = 'PuOr'
vmin, vmax = min(h_mean_posterior.min(), d_mean_e.min()), max(h_mean_posterior.max(), d_mean_e.max())

mat_mean_plot(h_mean_posterior, cmap=m, filename='hpost', vmin=vmin, vmax=vmax)

mat_mean_plot(d_cca_prediction.reshape(-1, 1) + d_modeling_mean_error - g @ h_mean,
              cmap=m, filename='mean', vmin=vmin, vmax=vmax)

# h_cca_prediction_gaussian = processing.gaussian_distribution(h_cca_prediction.T)
# mat_mean_plot(h_cca_prediction_gaussian.T,
#               cmap=m, filename='mean_post_gaussian', vmin=-1.649, vmax=1.590)
