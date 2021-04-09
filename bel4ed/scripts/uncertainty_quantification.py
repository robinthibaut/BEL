#  Copyright (c) 2021. Robin Thibaut, Ghent University
from os.path import join as jp

import pandas as pd
from loguru import logger
import numpy as np
from sklearn.model_selection import KFold, train_test_split

from bel4ed import init_bel
from bel4ed.algorithms import modified_hausdorff, structural_similarity
from bel4ed.config import Setup
from bel4ed.datasets import i_am_root, load_dataset
from bel4ed.design import bel_training, bel_uq
from bel4ed.goggles import mode_histo


def run(
        model,
        *,
        training_idx: list = None,
        test_idx: list = None,
        source_ids: list or np.array = None,
        kfold: bool = False,
        n_splits: int = None,
        train_size: int = 200,
        test_size: int = 50,
        random_state: int = None,
        shuffle: bool = False,
        name: str = None
):
    # Load datasets
    X, Y = load_dataset()

    if kfold:

        try:
            idx = np.array([*training_idx, *test_idx], dtype=object)
            X_ = X.loc[idx]
            y_ = Y.loc[idx]
        except TypeError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                Y,
                train_size=train_size,
                test_size=test_size,
                shuffle=shuffle,
                random_state=random_state,
            )

            idx = [*X_train.index, *X_test.index]
            X_ = pd.concat([X_train, X_test])
            X_.attrs["physical_shape"] = X.attrs["physical_shape"]
            y_ = pd.concat([y_train, y_test])
            y_.attrs["physical_shape"] = Y.attrs["physical_shape"]

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        kf.get_n_splits(idx)
        ns = 0  # Split number
        for train_index, test_index in kf.split(idx):
            logger.info(f"Fold {ns}")
            logger.info(train_index)
            logger.info(test_index)
            fold_directory = jp(
                Setup.Directories.forecasts_dir, f"{name}fold_{len(idx)}_{ns}"
            )

            X_train = X_.iloc[train_index]
            X_test = X_.iloc[test_index]
            y_train = y_.iloc[train_index]
            y_test = y_.iloc[test_index]

            index = X_test.index

            bel_training(
                bel=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                directory=fold_directory,
                source_ids=source_ids,
            )
            ns += 1

            # Pick metrics
            metrics = (modified_hausdorff,)

            # Compute UQ with metrics
            bel_uq(
                index=index,
                directory=fold_directory,
                source_ids=wells,
                metrics=metrics,
                delete=True,
                clear=True,
            )

            [plot_uq(m, directory=fold_directory) for m in metrics]

    elif training_idx and test_idx:
        if name:
            pre = name
        else:
            pre = "test"
        custom_directory = jp(
            Setup.Directories.forecasts_dir,
            f"{pre}_{len(training_idx)}_{len(test_idx)}",
        )
        # Select roots for testing
        X_train = X.loc[training_idx]
        X_test = X.loc[test_idx]
        y_train = Y.loc[training_idx]
        y_test = Y.loc[test_idx]

        bel_training(
            bel=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            directory=custom_directory,
            source_ids=source_ids,
        )

        # Pick metrics
        metrics = (structural_similarity,)

        # Compute UQ with metrics
        bel_uq(
            index=test_idx,
            directory=custom_directory,
            source_ids=wells,
            metrics=metrics,
        )

        [plot_uq(m, directory=custom_directory) for m in metrics]

    else:
        if name:
            pre = name
        else:
            pre = "test"
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
            source_ids=source_ids,
        )

        # Pick metrics
        metrics = (structural_similarity,)
        index = X_test.index
        # Compute UQ with metrics
        bel_uq(
            index=index,
            directory=test_directory,
            source_ids=wells,
            metrics=metrics,
            delete=True,
        )

        [plot_uq(m, directory=test_directory) for m in metrics]


def plot_uq(metric_function, directory: str = None):
    if directory is None:
        directory = Setup.Directories.forecasts_dir
    wm = np.load(jp(directory, f"uq_{metric_function.__name__}.npy"))
    colors = Setup.Wells.colors
    mode_histo(
        colors=colors,
        wm=wm,
        an_i=0,
        directory=directory,
        fig_name=metric_function.__name__,
    )


if __name__ == "__main__":
    # Get roots used for testing
    training_file = jp(Setup.Directories.storage_dir, "roots.dat")
    test_file = jp(Setup.Directories.storage_dir, "test_roots.dat")
    training_r, test_r = i_am_root(training_file=training_file, test_file=test_file)

    # Source IDs
    wells = np.array([[1], [2], [3], [4], [5], [6]], dtype=object)
    # wells = np.array([[1, 2, 3, 4, 5, 6]], dtype=object)

    # Initiate BEL model
    bel = init_bel()

    # Train model
    # Reference dataset
    # run(model=bel, training_idx=training_r, test_idx=test_r, source_ids=wells)
    # Test
    # idx_ = [*training_r, *test_r]
    # training_test = idx_[50:]
    # test_test = idx_[:50]
    # run(model=bel, training_idx=training_test, test_idx=test_test, source_ids=wells, name="check")

    # KFold on custom dataset
    run(
        model=bel,
        # training_idx=training_r,
        # test_idx=test_r,
        train_size=1000,
        test_size=250,
        source_ids=wells,
        kfold=True,
        n_splits=5,
        shuffle=False,
        random_state=None,
        name="new_K",
    )

    # Test datasets with various sizes
    # run(model=bel, source_ids=wells, train_size=1000, test_size=250, shuffle=True, random_state=6492, name="new")
    # run(model=bel, source_ids=wells, random_state=7017162, shuffle=True)
