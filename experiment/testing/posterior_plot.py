#  Copyright (c) 2021. Robin Thibaut, Ghent University

from experiment.visualization import kde_cca

if __name__ == '__main__':
    kde_cca(root='818bf1676c424f76b83bd777ae588a1d',
            well='123456',
            comp_n=0,
            show=True,
            dist_plot=True)
