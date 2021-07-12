#  Copyright (c) 2021. Robin Thibaut, Ghent University
from os.path import join as jp

import multiprocessing as mp

import numpy as np

from sklearn.model_selection import KFold, train_test_split

from bel4ed import kernel_bel, init_bel
from bel4ed.config import Setup
from bel4ed.datasets import i_am_root, load_dataset
from bel4ed.design import bel_training, bel_uq, bel_training_mp
from bel4ed.design import bel_uq_mp
from bel4ed.design import find_extreme, plot_uq

from bel4ed.metrics import modified_hausdorff, structural_similarity


if __name__ == "__main__":
    wells_training = np.array(
        [[1, 2, 3, 4, 5, 6], [1], [2], [3], [4], [5], [6]], dtype=object
    )
    wells_uq = np.array([[1], [2], [3], [4], [5], [6]], dtype=object)
    name = "aniso"
    # Initiate BEL model
    bel = init_bel()
    bel.n_posts = Setup.HyperParameters.n_posts
    bel.mode = "mvn"
    bel.X_shape = (6, 200)
    bel.Y_shape = (1, 80, 110)

    # Load datasets
    X, Y = load_dataset(subdir="data_structural")

    # random_state = np.random.randint(0, 1e7)
    # random_state = 648908
    random_state = 3529748

    train_size = 1000
    test_size = 250
    shuffle = True

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        train_size=train_size,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
    )

    test_roots = [t for t in X_test.index]
    test_directory = jp(
        Setup.Directories.forecasts_dir,
        f"{name}_{train_size}_{test_size}_{random_state}",
    )

    train_roots = [t for t in X_train.index]

    with open(
        jp(
            Setup.Directories.storage_dir,
            f"training_roots_{name}_{train_size}_{random_state}.dat",
        ),
        "w",
    ) as doc:
        for line in train_roots:
            doc.write(line + "\n")
    test_roots = [t for t in X_test.index]

    with open(
        jp(
            Setup.Directories.storage_dir,
            f"test_roots_{name}_{test_size}_{random_state}.dat",
        ),
        "w",
    ) as doc:
        for line in test_roots:
            doc.write(line + "\n")

    test_directory = jp(
        Setup.Directories.forecasts_dir,
        f"{name}_{train_size}_{test_size}_{random_state}",
    )

    args = [
        (bel, X_train, X_test, y_train, y_test, test_directory, wells_training, tr)
        for tr in test_roots
    ]

    # n_cpu = 8
    # pool = mp.Pool(n_cpu)
    # pool.map(bel_training_mp, args)
    # pool.close()
    # pool.join()

    # Pick metrics
    metrics = (
        modified_hausdorff,
        structural_similarity,
    )
    index = X_test.index
    argsuq = [(bel, y_test, tr, test_directory, wells_uq, metrics) for tr in test_roots]
    # Compute UQ with metrics
    # n_cpu = 12
    # pool = mp.Pool(n_cpu)
    # pool.map(bel_uq_mp, argsuq)
    # pool.close()
    # pool.join()

    names = ["MHD", "SSIM"]

    # [
    #     plot_uq(
    #         bel,
    #         index,
    #         m,
    #         directory=test_directory,
    #         combi=wells_uq,
    #         title=f"{names[ix]} Training/Test {len(X_train)}/{len(X_test)}",
    #         an_i=ix,
    #     )
    #     for ix, m in enumerate(metrics)
    # ]
    #
    root, comb, mxr, mxc = find_extreme(
        index, modified_hausdorff, [(1, 2, 3, 4, 5, 6)], test_directory
    )
