#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os

import matplotlib.pyplot as plt
import numpy as np
import string
import seaborn as sns
from typing import List

from experiment.base.inventory import MySetup

from experiment.goggles.visualization import Plot, proxy_annotate

Root = List[str]


def value_info(root: Root):
    """
    Computes the combined value of information for n observations.
    see also
    https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
    :param root: list: List containing the roots whose wells contributions will be taken into account.
    :return:
    """

    alphabet = string.ascii_uppercase

    if not isinstance(root, (list, tuple)):
        root: list = [root]

    # Deals with the fact that only one root might be selected
    fig_name = 'average'
    an_i = 0  # Annotation index
    if len(root) == 1:
        fig_name = root[0]
        an_i = 2

    wid = list(map(str, MySetup.Wells.combination))  # Wel identifiers (n)
    wm = np.zeros((len(wid), MySetup.Forecast.n_posts))  # Summed MHD when well #i appears

    for r in root:  # For each root
        droot = os.path.join(MySetup.Directories.forecasts_dir, r)  # Starting point = root folder in forecast directory
        for e in wid:  # For each subfolder (well) in the main folder
            fmhd = os.path.join(droot, e, 'obj', 'haus.npy')  # Get the MHD file
            mhd = np.load(fmhd)  # Load MHD
            idw = int(e) - 1  # -1 to respect 0 index (Well index)
            wm[idw] += mhd  # Add MHD at each well

    colors = Plot().cols  # Get default colors from visualization class

    modes = []  # Get MHD corresponding to each well's mode
    for i, m in enumerate(wm):  # For each well, look up its MHD distribution
        count, values = np.histogram(m)
        idm = np.argmax(count)
        mode = values[idm]
        modes.append(mode)

    # TODO: Put visualization methods in proper folder
    modes = np.array(modes)  # Scale modes
    modes -= np.mean(modes)

    # Bar plot
    plt.bar(np.arange(1, 7), -modes, color=colors)
    plt.title('Value of information of each well')
    plt.xlabel('Well ID')
    plt.ylabel('Opposite deviation from mode\'s mean')
    plt.grid(color='#95a5a6', linestyle='-', linewidth=.5, axis='y', alpha=0.7)

    legend_a = proxy_annotate(annotation=[alphabet[an_i+1]], loc=2, fz=14)
    plt.gca().add_artist(legend_a)

    plt.savefig(os.path.join(MySetup.Directories.forecasts_dir, f'{fig_name}_well_mode.pdf'), dpi=300, transparent=True)
    plt.close()
    # plt.show()

    # Plot histogram
    for i, m in enumerate(wm):
        sns.kdeplot(m, color=f'{colors[i]}', shade=True, linewidth=2)
    plt.title('Summed MHD distribution for each well')
    plt.xlabel('Summed MHD')
    plt.ylabel('KDE')
    legend_1 = plt.legend(wid, loc=1)
    plt.gca().add_artist(legend_1)
    plt.grid(alpha=0.2)

    legend_a = proxy_annotate(annotation=[alphabet[an_i]], loc=2, fz=14)
    plt.gca().add_artist(legend_a)

    plt.savefig(os.path.join(MySetup.Directories.forecasts_dir, f'{fig_name}_hist.pdf'), dpi=300, transparent=True)
    plt.close()
    # plt.show()

    # %% Facet histograms
    # ids = np.array(np.concatenate([np.ones(wm.shape[1]) * i for i in range(1, 7)]), dtype='int')
    # master = wm.flatten()
    #
    # data = np.concatenate([[master], [ids]], axis=0)
    #
    # master_x = pd.DataFrame(data=data.T, columns=['MHD', 'well'])
    # master_x['well'] = np.array(ids)
    # g = sns.FacetGrid(master_x,  # the dataframe to pull from
    #                   row="well",
    #                   hue="well",
    #                   aspect=3,  # aspect * height = width
    #                   height=1.5,  # height of each subplot
    #                   palette=colors  # google colors
    #                   )
    #
    # g.map(sns.kdeplot, "MHD", shade=True, alpha=1, lw=1.5)
    # g.map(plt.axhline, y=0, lw=4)
    # for ax in g.axes:
    #     ax[0].set_xlim((500, 1000))
    #
    # def label(x, color, label):
    #     ax = plt.gca()  # get the axes of the current object
    #     ax.text(0, .2,  # location of text
    #             label,  # text label
    #             fontweight="bold", color=color, size=20,  # text attributes
    #             ha="left", va="center",  # alignment specifications
    #             transform=ax.transAxes)  # specify axes of transformation)
    #
    # g.map(label, "MHD")  # the function counts as a plotting object!
    #
    # sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # g.fig.subplots_adjust(hspace=-.25)
    #
    # g.set_titles("")  # set title to blank
    # g.set_xlabels(color="white")
    # g.set_xticklabels(color='white', fontsize=14)
    # g.set(yticks=[])  # set y ticks to blank
    # g.despine(bottom=True, left=True)  # remove 'spines'
    #
    # plt.savefig(os.path.join(MySetup.Directories.forecasts_dir, f'{fig_name}_facet.pdf'), dpi=300, transparent=True)
    # plt.close()
    # plt.show()


if __name__ == '__main__':
    # Value info
    forecast_dir = MySetup.Directories.forecasts_dir
    listit = os.listdir(forecast_dir)
    listit.remove('base')
    duq = list(filter(lambda f: os.path.isdir(os.path.join(forecast_dir, f)), listit))  # Folders of combinations

    value_info(duq)
    value_info(['818bf1676c424f76b83bd777ae588a1d'])
