#  Copyright (c) 2020. Robin Thibaut, Ghent University
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as jp
import joblib

import seaborn as sb

from sklearn.decomposition import PCA


from experiment.goggles.visualization import Plot

from experiment.base.inventory import MySetup

md = MySetup.Directories()

root = '6a4d614c838442629d7a826cc1f498a8'
sources = '123456'
sdir = jp(md.forecasts_dir, root, sources)
# post_obj = joblib.load(jp(sdir, 'obj', 'post.pkl'))
# h_samples = post_obj.random_sample()

sb.set_theme()

ndor = jp(md.forecasts_dir, 'base', 'roots_whpa', f'{root}.npy')
nn = np.load(ndor)
nnt = np.flipud(nn[0])
plt.imshow(nnt)
plt.colorbar()
plt.show()

pca = PCA(n_components=20)
pca.fit(nnt)
scores = pca.transform(nnt)

inv = pca.inverse_transform(scores).reshape(100, 87)
plt.imshow(inv)
plt.colorbar()
plt.show()


#%%

fc = MySetup.Focus()
x_lim, y_lim, grf = fc.x_range, fc.y_range, fc.cell_dim
mplot = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf, well_comb=None)
mplot.whp(h=nn,
          x_lim=x_lim,
          y_lim=[335, 700],
          labelsize=11,
          xlabel='X(m)',
          ylabel='Y(m)',
          cb_title='SD(m)',
          bkg_field_array=np.flipud(nn[0]),
          colors='black',
          cmap=None)
# plt.savefig(jp(md.forecasts_dir, 'base', 'roots_whpa', f'{root}_SD.pdf'),
#             bbox_inches='tight',
#             dpi=300, transparent=True)
plt.show()
