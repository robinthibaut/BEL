#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
import string

from experiment.base.inventory import MySetup
import experiment.goggles.visualization as myvis
from experiment.toolbox.filesio import data_read


if __name__ == '__main__':

    base_dir = os.path.join(MySetup.Directories.forecasts_dir, 'base')
    test_roots = data_read(os.path.join(base_dir, 'test_roots.dat'))
    samples = [item for sublist in test_roots for item in sublist]

    # roots = samples
    #
    roots = ['818bf1676c424f76b83bd777ae588a1d',
             'dc996e54728b4bb4a7234ee691729076',
             '27ec76adab2e406794584fc993188c24',
             '9a389395bfbe4cd883dfa3e452752978']

    # roots = ['818bf1676c424f76b83bd777ae588a1d']

    alphabet = string.ascii_uppercase

    fc = MySetup.Focus()
    x_lim, y_lim, grf = fc.x_range, fc.y_range, fc.cell_dim

    # ['123456', '1', '2', '3', '4', '5', '6']

    for i, sample in enumerate(roots):
        print(f'Plotting root {sample}')

        wells = ['123456', '1', '2', '3', '4', '5', '6']

        for j, w in enumerate(wells):

            print(f'Plotting well {w}')

            if w == '123456':
                annotation = alphabet[i]
            else:
                annotation = alphabet[j-1]

            myvis.plot_results(root=sample,
                               folder=w,
                               annotation=annotation,
                               d=False)

        myvis.plot_K_field(root=sample)

        myvis.plot_head_field(root=sample)

        myvis.plot_whpa(root=sample)
        #
        myvis.pca_vision(root=sample,
                         d=True,
                         h=True,
                         exvar=True,
                         labels=True,
                         scores=True,
                         folders=wells)

        myvis.plot_pc_ba(root=sample,
                         data=True,
                         target=True)

        myvis.cca_vision(root=sample,
                         folders=wells)
        #
