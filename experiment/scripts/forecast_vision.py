#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os

from experiment.base.inventory import MySetup
from experiment.goggles.visualization import Plot
from experiment.toolbox.filesio import datread, empty_figs


def main(roots):
    fc = MySetup.Focus()
    x_lim, y_lim, grf = fc.x_range, fc.y_range, fc.cell_dim
    mplot = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf, wel_comb=None)
    for sample in roots:
        print(f'Plotting root {sample}')
        # plot_pc_ba(sample, data=True, target=True)
        # empty_figs(sample)
        wells = ['123456', '1', '2', '3', '4', '5', '6']
        for w in wells:
            print(f'Plotting well {w}')
            mplot.plot_results(root=sample, folder=w)
        mplot.plot_whpa(sample)
        mplot.cca_vision(sample, folders=wells)
        mplot.pca_vision(sample, d=True, h=True, exvar=True, scores=True,
                         folders=wells)
        # ['123456', '1', '2', '3', '4', '5', '6']


if __name__ == '__main__':
    base_dir = os.path.join(MySetup.Directories.forecasts_dir, 'base')
    test_roots = datread(os.path.join(base_dir, 'test_roots.dat'))
    samples = [item for sublist in test_roots for item in sublist]
    main(['bb3d985d33cf44a3b779fa8a40a418d0'])


