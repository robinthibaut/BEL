#  Copyright (c) 2020. Robin Thibaut, Ghent University
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as jp
import joblib
from experiment.goggles.visualization import Plot

from experiment.base.inventory import MySetup

md = MySetup.Directories()

root = '6a4d614c838442629d7a826cc1f498a8'
sources = '123456'
sdir = jp(md.forecasts_dir, root, sources)
# post_obj = joblib.load(jp(sdir, 'obj', 'post.pkl'))
# h_samples = post_obj.random_sample()

ndor = jp(md.forecasts_dir, 'base', 'roots_whpa', f'{root}.npy')
nn = np.load(ndor)
plt.imshow(nn[0])
plt.colorbar()
plt.show()

fc = MySetup.Focus()
x_lim, y_lim, grf = fc.x_range, fc.y_range, fc.cell_dim
mplot = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf, wel_comb=None)
mplot.whp(h=nn, bkg_field_array=np.flipud(nn[0]), colors='black', cmap=None)
plt.savefig(jp(md.forecasts_dir, 'base', 'roots_whpa', f'{root}_SD.png'), bbox_inches='tight', dpi=300, transparent=True)
plt.show()