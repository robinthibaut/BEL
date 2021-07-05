#  Copyright (c) 2021. Robin Thibaut, Ghent University
from os.path import join as jp

import numpy as np
import numpy.random
import pandas as pd
from loguru import logger
from skbel.utils import combinator
from sklearn.model_selection import KFold, train_test_split

from bel4ed import kernel_bel
from bel4ed.config import Setup
from bel4ed.datasets import i_am_root, load_dataset
from bel4ed.design import bel_training, bel_uq
from bel4ed.goggles import mode_histo
from bel4ed.metrics import modified_hausdorff, structural_similarity


def run(
        model,
        *,
        source_ids_training: list or np.array = None,
        source_ids_uq: list or np.array = None,
        train_size: int = 200,
        test_size: int = 50,
        random_state: int = None,
        shuffle: bool = False,
        name: str = None,
):
    # Load datasets
    X, Y = load_dataset()

    if source_ids_uq is None:
        source_ids_uq = source_ids_training

    if name:
        pre = name
    else:
        pre = "test"
    if random_state is None:
        random_state = np.random.randint(0, 1e7)
    test_directory = jp(
        Setup.Directories.forecasts_dir,
        f"{pre}_{train_size}_{test_size}_{random_state}",
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        train_size=train_size,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
    )

    bel_training(
        bel=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        directory=test_directory,
        source_ids=source_ids_training,
    )

    # Pick metrics
    metrics = (
        # modified_hausdorff,
        structural_similarity,
    )
    index = X_test.index
    # Compute UQ with metrics
    bel_uq(
        bel=model,
        index=index,
        directory=test_directory,
        source_ids=source_ids_uq,
        metrics=metrics,
        delete=False,
    )

    try:
        [
            plot_uq(
                m,
                directory=test_directory,
                title=f"{m.__name__.capitalize()} Training/Test {len(X_train)}/{len(X_test)}",
                an_i=ix,
            )
            for ix, m in enumerate(metrics)
        ]
    except ValueError:
        pass


def plot_uq(metric_function, directory: str = None, title: str = None, an_i: int = 0):
    if directory is None:
        directory = Setup.Directories.forecasts_dir
    wm = np.load(jp(directory, f"uq_{metric_function.__name__}.npy"))
    colors = Setup.Wells.colors
    mode_histo(
        colors=colors,
        wm=wm,
        an_i=an_i,
        title=title,
        directory=directory,
        fig_name=metric_function.__name__,
    )


if __name__ == "__main__":
    # %%
    # Source IDs
    wells_uq = np.array([[1], [2], [3], [4], [5], [6]], dtype=object)
    wells_training = np.array([[1], [2], [3], [4], [5], [6]], dtype=object)

    combis = combinator([1, 2, 3, 4, 5, 6])
    c23 = list(filter(lambda t: 1 < len(t) < 4, combis))
    # %%
    # Initiate BEL model
    bel = kernel_bel()
    bel.n_posts = Setup.HyperParameters.n_posts
    bel.mode = "kde"
    bel.X_shape = (6, 200)  # Six curves with 200 time steps each
    bel.Y_shape = (1, 100, 87)  # One matrix with 100 rows and 87 columns
    # Train model

    # %%
    run(model=bel,
        source_ids_training=c23,
        source_ids_uq=c23,
        train_size=1000,
        test_size=250,
        shuffle=True)
