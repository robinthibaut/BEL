#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from experiment.base.inventory import MySetup


def dists(res_dir, folders=None):
    subdir = os.path.join(MySetup.Directories.forecasts_dir, res_dir)
    if folders is None:
        listme = os.listdir(subdir)
        folders = list(filter(lambda du: os.path.isdir(os.path.join(subdir, du)), listme))
    else:
        if not isinstance(folders, (list, tuple)):
            folders = [folders]

    obj_ = []

    for f in folders:
        # For d only
        pcnpy = os.path.join(subdir, f, 'uq', '500_target_pc.npy')
        h_pc = np.load(pcnpy)
        obj_.append(h_pc)

    return obj_


sample = '6623dd4fb5014a978d59b9acb03946d2'
default = ['123456']

dpc_post = dists(sample, folders=default)[0]
hp1 = dpc_post[:200, 0]

# %%

hbase = os.path.join(MySetup.Directories.forecasts_dir, 'base')
# Load h pickle
pcaf = os.path.join(hbase, 'h_pca.pkl')
h_pco = joblib.load(pcaf)

h1 = h_pco.training_pc[:, 0]

# %%
# Plot kde distribution
wm = [h1, hp1]
colors = ['blue', 'orange']

for i, m in enumerate(wm):
    sns.kdeplot(m, color=f'{colors[i]}', shade=True, linewidth=2)
plt.title('First PC distribution of target prior and posterior')
plt.xlabel('First PC score')
plt.ylabel('KDE')
plt.legend(['Prior', 'Posterior'])
plt.grid(alpha=0.2)
# plt.savefig(os.path.join(MySetup.Directories.forecasts_dir, f'{sample}_hist.png'), dpi=300, transparent=True)
plt.show()

sns.distplot(h1, bins=20, kde=False)
sns.distplot(hp1, bins=20, kde=False)
plt.title('First PC distribution of target prior and posterior')
plt.xlabel('First PC score')
plt.ylabel('Count')
plt.legend(['Prior', 'Posterior'])
plt.grid(alpha=0.2)
# plt.savefig(os.path.join(MySetup.Directories.forecasts_dir, f'{sample}_hist.png'), dpi=300, transparent=True)
plt.show()
plt.show()