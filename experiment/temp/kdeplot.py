#  Copyright (c) 2021. Robin Thibaut, Ghent University

import numpy as np

import matplotlib.pyplot as plt

from experiment._statistics import posterior_conditional

if __name__ == '__main__':
    # Generate some 2D data
    rs = np.random.RandomState(5)
    mean = [0, 0]
    cov = [(1, .98), (.98, 1)]
    predictor, target = rs.multivariate_normal(mean, cov, 100).T

    mean2 = [1, 1]
    cov2 = [(1, -.98), (-.98, 1)]
    predictor2, target2 = rs.multivariate_normal(mean2, cov2, 100).T

    predictor = np.concatenate((predictor, predictor2), axis=0)
    target = np.concatenate((target, target2), axis=0)

    conditional_value = -0.5
    h_post, sup = posterior_conditional(predictor, target, conditional_value)

    plt.plot(predictor, target, 'ko')
    plt.show()
    plt.plot(h_post)
    plt.show()
