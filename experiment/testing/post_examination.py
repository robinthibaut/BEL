#  Copyright (c) 2021. Robin Thibaut, Ghent University
from os.path import join as jp

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

import experiment._visualization as mplot
from experiment._core import Setup

md = Setup.Directories

root = '818bf1676c424f76b83bd777ae588a1d'
sources = '123456'
sdir = jp(md.forecasts_dir, root, sources)
# post_obj = joblib.load(jp(sdir, 'obj', 'post.pkl'))
# h_samples = post_obj.random_sample()

sb.set_theme()

ndor = jp(md.forecasts_dir, 'base', 'roots_whpa', f'{root}.npy')
nn = np.load(ndor)
# nnt = np.flipud(nn[0])
# plt.imshow(nnt)
# plt.colorbar()
# plt.show()

# pca = PCA(n_components=20)
# pca.fit(nnt)
# scores = pca.transform(nnt)

# inv = pca.inverse_transform(scores).reshape(100, 87)
# plt.imshow(inv)
# plt.colorbar()
# plt.show()


# %%

fc = Setup.Focus
x_lim, y_lim, grf = fc.x_range, fc.y_range, fc.cell_dim

mplot.whpa_plot(whpa=nn,
                x_lim=x_lim,
                y_lim=[335, 700],
                labelsize=11,
                alpha=1,
                xlabel='X(m)',
                ylabel='Y(m)',
                cb_title='SD(m)',
                annotation=['B'],
                bkg_field_array=np.flipud(nn[0]),
                color='black')

# legend = proxy_annotate(annotation=['B'], loc=2, fz=14)
# plt.gca().add_artist(legend)

plt.savefig(jp(md.forecasts_dir, 'base', 'roots_whpa', f'{root}_SD.pdf'),
            bbox_inches='tight',
            dpi=300, transparent=True)
plt.show()
