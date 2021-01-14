#  Copyright (c) 2020. Robin Thibaut, Ghent University
import experiment.toolbox.filesio as fops
import numpy as np
import matplotlib.pyplot as plt
from experiment.goggles.visualization import proxy_annotate

from os.path import join as jp
from experiment.base.inventory import MySetup
md = MySetup.Directories()

# model_nam = '/Users/robin/PycharmProjects/BEL/experiment/storage/forwards/6a4d614c838442629d7a826cc1f498a8/whpa.nam'
#
# flow = fops.load_flow_model(model_nam)

root = '818bf1676c424f76b83bd777ae588a1d'
ep = f'/Users/robin/PycharmProjects/BEL/experiment/storage/forwards/{root}/tracking_ep.npy'

epxy = np.load(ep)

plt.close()
plt.plot(epxy[:, 0], epxy[:, 1], 'ko')
# seed = np.random.randint(2**32 - 1)
# np.random.seed(seed)
# sample = np.random.randint(144, size=10)
sample = np.array([94, 10, 101,  29,  43, 116, 100,  40,  72])
for i in sample:
    plt.text(epxy[i, 0] + 4, epxy[i, 1] + 4, i, color='black', fontsize=11, weight='bold',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=.5', alpha=.7))
    # plt.annotate(i, (epxy[i, 0] + 4, epxy[i, 1] + 4), fontsize=14, weight='bold', color='r')

plt.grid(alpha=.5)
plt.xlim([870, 1080])
plt.ylim([415, 600])
plt.xlabel('X(m)')
plt.ylabel('Y(m)')
plt.tick_params(labelsize=11)

legend = proxy_annotate(annotation=['A'], loc=2, fz=14)
plt.gca().add_artist(legend)

plt.savefig(jp(md.forecasts_dir, 'base', 'roots_whpa', f'{root}_ep.pdf'),
            dpi=300, bbox_inches='tight', transparent=True)
plt.show()
# print(seed)

# 4088225279
# 1440052516 but remove 60

#%%

# test_array = epxy
#
# mypca = PCA()
# mypca.fit(test_array)
#
# scores = mypca.transform(test_array)
#
# test = mypca.inverse_transform(scores).reshape(144, 2)
# plt.close()
# plt.plot(test, 'ko')
# plt.show()
