#  Copyright (c) 2021. Robin Thibaut, Ghent University

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
# plt.style.use('dark_background')

from experiment.base.inventory import MySetup as base
from experiment.goggles.visualization import Plot
from experiment.toolbox import filesio as fops

x_lim, y_lim, grf = base.focus.x_range, base.focus.y_range, base.focus.cell_dim
# Initiate Plot instance
mp = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf, well_comb=base.wells.combination)

tc0, sd, r = fops.data_loader(res_dir=base.directories.hydro_res_dir,
                              roots=['6a4d614c838442629d7a826cc1f498a8'],
                              d=True, h=True)

# for i, w in enumerate(base.Wells.combination):
#     plt.plot(tc0[0][i][:, 0], tc0[0][i][:, 1], mp.cols[i])
# plt.show()

# plt.scatter(sd[0, :, 0], sd[0, :, 1], s=1)
# plt.show()

xy = {'X(m)': sd[0, :, 0], 'Y(m)': sd[0, :, 1]}
data = pd.DataFrame(xy)

plt.plot(sd[0, :, 0], sd[0, :, 1], 'k', linewidth=.5)
sns.scatterplot(x='X(m)', y='Y(m)', data=data)
plt.show()
