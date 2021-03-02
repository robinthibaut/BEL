#  Copyright (c) 2021. Robin Thibaut, Ghent University

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import experiment.utils
from experiment import _utils as fops
from experiment.core import Setup

sns.set_theme()
# plt.style.use('dark_background')

x_lim, y_lim, grf = Setup.Focus.x_range, Setup.Focus.y_range, Setup.Focus.cell_dim
# Initiate Plot instance

tc0, sd, r = experiment.utils.data_loader(
    res_dir=Setup.Directories.hydro_res_dir,
    roots=["6a4d614c838442629d7a826cc1f498a8"],
    d=True,
    h=True,
)

# for i, w in enumerate(base.Wells.combination):
#     plt.plot(tc0[0][i][:, 0], tc0[0][i][:, 1], mp.cols[i])
# plt.show()

# plt.scatter(sd[0, :, 0], sd[0, :, 1], s=1)
# plt.show()

xy = {"X(m)": sd[0, :, 0], "Y(m)": sd[0, :, 1]}
data = pd.DataFrame(xy)

plt.plot(sd[0, :, 0], sd[0, :, 1], "k", linewidth=0.5)
sns.scatterplot(x="X(m)", y="Y(m)", data=data)
plt.show()
