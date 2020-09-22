#  Copyright (c) 2020. Robin Thibaut, Ghent University

import numpy as np
import matplotlib.pyplot as plt

from experiment.base.inventory import MySetup as base
from experiment.goggles.visualization import Plot
from experiment.toolbox import filesio as fops

x_lim, y_lim, grf = base.Focus.x_range, base.Focus.y_range, base.Focus.cell_dim
# Initiate Plot instance
mp = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf, wel_comb=base.Wells.combination)

tc0, sd, r = fops.data_loader(res_dir=base.Directories.hydro_res_dir,
                              roots=['1d596ce96cbd4b1681bc4423253f40e2'],
                              d=True, h=True)

for i, w in enumerate(base.Wells.combination):
    plt.plot(tc0[0][i][:, 0], tc0[0][i][:, 1], mp.cols[i])
plt.show()

plt.scatter(sd[0, :, 0], sd[0, :, 1])
plt.show()
