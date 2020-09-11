#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os

from experiment.base.inventory import MySetup
from experiment.goggles.visualization import Plot
from experiment.toolbox.filesio import datread, empty_figs


if __name__ == '__main__':

    base_dir = os.path.join(MySetup.Directories.forecasts_dir, 'base')
    test_roots = datread(os.path.join(base_dir, 'test_roots.dat'))
    samples = [item for sublist in test_roots for item in sublist]
    roots = ['46d0170062654fc3b36888f2e2510fcb']

    fc = MySetup.Focus()
    x_lim, y_lim, grf = fc.x_range, fc.y_range, fc.cell_dim
    mplot = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf, wel_comb=None)

    mplot.plot_pc_ba(roots[0], data=True, target=True)

    for sample in roots:
        print(f'Plotting root {sample}')
        # empty_figs(sample)
        wells = ['123456']
        for w in wells:
            print(f'Plotting well {w}')
            mplot.plot_results(root=sample, folder=w)
        mplot.plot_whpa(sample)
        mplot.cca_vision(sample, folders=wells)
        mplot.pca_vision(sample, d=True, h=True, exvar=True, scores=True,
                         folders=wells)
        # ['123456', '1', '2', '3', '4', '5', '6']
