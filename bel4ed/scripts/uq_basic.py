#  Copyright (c) 2021. Robin Thibaut, Ghent University
from os.path import join as jp

import multiprocessing as mp

import numpy as np
import numpy.random
import pandas as pd
from loguru import logger
from skbel.utils import combinator
from sklearn.model_selection import KFold, train_test_split

from bel4ed import kernel_bel, init_bel
from bel4ed.config import Setup
from bel4ed.datasets import i_am_root, load_dataset
from bel4ed.design import bel_training, bel_uq, bel_training_mp
from bel4ed.goggles import mode_histo
from bel4ed.metrics import modified_hausdorff, structural_similarity


def plot_uq(
    metric_function,
    combi: list = None,
    directory: str = None,
    title: str = None,
    an_i: int = 0,
):
    if directory is None:
        directory = Setup.Directories.forecasts_dir
    wm = np.load(jp(directory, f"uq_{metric_function.__name__}.npy"))
    colors = Setup.Wells.colors
    mode_histo(
        colors=colors,
        wm=wm,
        combi=combi,
        an_i=an_i,
        title=title,
        directory=directory,
        fig_name=metric_function.__name__,
    )


if __name__ == "__main__":
    wells_training = np.array(
        [[1, 2, 3, 4, 5, 6], [1], [2], [3], [4], [5], [6]], dtype=object
    )
    wells_uq = np.array([[1], [2], [3], [4], [5], [6]], dtype=object)
    name = "basic"
    # Initiate BEL model
    bel = init_bel()
    bel.n_posts = Setup.HyperParameters.n_posts
    bel.mode = "mvn"
    bel.X_shape = (6, 200)
    bel.Y_shape = (1, 100, 87)

    # Load datasets
    X, Y = load_dataset()

    random_state = np.random.randint(0, 1e7)

    train_size = 400
    test_size = 100
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

    args = [
        (bel, X_train, X_test, y_train, y_test, test_directory, wells_training, tr)
        for tr in test_roots
    ]
    n_cpu = 8
    pool = mp.Pool(n_cpu)
    pool.map(bel_training_mp, args)
    pool.close()
    pool.join()

    # Pick metrics
    metrics = (
        # modified_hausdorff,
        structural_similarity,
    )
    index = X_test.index
    # Compute UQ with metrics
    bel_uq(
        bel=bel,
        y_obs=y_test,
        index=index,
        directory=test_directory,
        source_ids=wells_uq,
        metrics=metrics,
        delete=True,
    )

    [
        plot_uq(
            m,
            directory=test_directory,
            combi=wells_uq,
            # title=f"{m.__name__.capitalize()} Training/Test {len(X_train)}/{len(X_test)}",
            an_i=ix,
        )
        for ix, m in enumerate(metrics)
    ]
