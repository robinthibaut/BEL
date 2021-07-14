#  Copyright (c) 2021. Robin Thibaut, Ghent University

from skbel.goggles import _my_alphabet, _proxy_annotate
from sklearn.cross_decomposition import CCA

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA, KernelPCA

n_samples = 1000

noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)


# Anisotropicly distributed data
random_state = 9
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.1, -0.8], [-0.2, 0.4]]
X_aniso = np.dot(X, transformation)
aniso2 = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

X_strange = np.column_stack(
    (aniso2[0][:, 0], noisy_circles[0][:, 0], noisy_moons[0][:, 0], aniso[0][:, 0], varied[0][:, 0]))
y_strange = np.column_stack(
    (aniso2[0][:, 1], noisy_circles[0][:, 1], noisy_moons[0][:, 1], aniso[0][:, 1], varied[0][:, 1]))

for i in range(X_strange.shape[1]):
    plt.plot(X_strange[:, i], y_strange[:, i], "o")
    plt.xlabel(f"X {i+1}")
    plt.ylabel(f"Y {i+1}")
    annotation = _my_alphabet(i)
    _proxy_annotate(annotation=annotation, loc=2)
    # plt.savefig(f"XY{i}.png", bbox_inches="tight", dpi=300, transparent=False)
    plt.show()

#
# pca = KernelPCA(kernel="rbf")
#
# xpc, ypc = (pca.fit_transform(X_strange), pca.fit_transform(y_strange))

xpc, ypc = X_strange, y_strange

ncomp = 4
cca = CCA(n_components=ncomp)

xcv, ycv = cca.fit_transform(xpc, ypc)

for i in range(ncomp):
    plt.plot(xcv[:, i], ycv[:, i], "o")
    plt.xlabel(f"X CV {i+1}")
    plt.ylabel(f"Y CV {i+1}")
    annotation = _my_alphabet(i)
    _proxy_annotate(annotation=annotation, loc=2)
    # plt.savefig(f"CV{i}.png", bbox_inches="tight", dpi=300, transparent=True)
    plt.show()
