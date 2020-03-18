import numpy as np
from numpy.matlib import repmat


def posterior(h_cca_training_gaussian, d_cca_training, d_pc_training, d_rotations, d_cca_prediction):
    """
    Estimating posterior uncertainties.
    @param h_cca_training_gaussian:
    @param d_cca_training:
    @param d_pc_training:
    @param d_rotations:
    @param d_cca_prediction:
    @return: h_mean_posterior, h_posterior_covariance
    """

    # TODO: add dimension check

    n_training = d_cca_training.shape[1]
    # Evaluate the covariance in h
    h_cov_operator = np.cov(h_cca_training_gaussian.T, rowvar=False)  # same

    # Evaluate the covariance in d (here we assume no data error, so C is identity times a given factor)
    x_dim = np.size(d_pc_training, axis=1)  # Number of PCA components for the curves
    noise = .01
    d_cov_operator = np.eye(x_dim) * noise  # I matrix.
    d_noise_covariance = d_rotations.T @ d_cov_operator @ d_rotations  # same

    # Linear modeling d to h
    g = np.linalg.lstsq(h_cca_training_gaussian.T, d_cca_training.T, rcond=None)[0].T  # Transpose to get same as Thomas
    g = np.where(np.abs(g) < 1e-12, 0, g)

    # Modeling error due to deviations from theory
    d_ls_predicted = g @ h_cca_training_gaussian  # same
    d_modeling_mean_error = np.mean(d_cca_training - d_ls_predicted, axis=1).reshape(-1, 1)  # same
    d_modeling_error = d_cca_training - d_ls_predicted - repmat(d_modeling_mean_error, 1,
                                                                np.size(d_cca_training, axis=1))  # same
    # Information about the covariance of the posterior distribution.
    d_modeling_covariance = (d_modeling_error @ d_modeling_error.T) / n_training  # same

    # Computation of the posterior mean
    h_mean = np.row_stack(np.mean(h_cca_training_gaussian, axis=1))  # same
    h_mean = np.where(np.abs(h_mean) < 1e-12, 0, h_mean)  # My mean is 0, as expected.

    h_mean_posterior = h_mean \
                       + h_cov_operator @ g.T \
                       @ np.linalg.pinv(g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance) \
                       @ (d_cca_prediction.reshape(-1, 1) + d_modeling_mean_error - g @ h_mean)  # same

    # h posterior covariance
    h_posterior_covariance = h_cov_operator \
                             - (h_cov_operator @ g.T) \
                             @ np.linalg.pinv(g @ h_cov_operator @ g.T + d_noise_covariance + d_modeling_covariance) \
                             @ g @ h_cov_operator

    h_posterior_covariance = (h_posterior_covariance + h_posterior_covariance.T) / 2  # same

    return h_mean_posterior, h_posterior_covariance
