#  Copyright (c) 2021. Robin Thibaut, Ghent University
from os.path import join as jp

import numpy as np
from sklearn.model_selection import KFold

from bel4ed import init_bel
from bel4ed.algorithms import modified_hausdorff, structural_similarity
from bel4ed.config import Setup
from bel4ed.datasets import i_am_root, load_dataset
from bel4ed.design import bel_training, bel_uq
from bel4ed.goggles import mode_histo


def run(model,
        training_idx: list, test_idx: list,
        source_ids: list or np.array,
        kfold: bool = False, n_splits: int = None):
    # Load datasets
    X, Y = load_dataset()

    if kfold:
        idx = [*training_idx, *test_idx]

        X_ = X.loc[idx]
        y_ = Y.loc[idx]

        kf = KFold(n_splits=n_splits)
        kf.get_n_splits(X_)
        ns = 0  # Split number
        for train_index, test_index in kf.split(X_):
            fold_directory = jp(Setup.Directories.forecasts_dir, f"fold_{ns}")
            bel_training(
                bel=model,
                X_train=X_.iloc[train_index],
                X_test=X_.iloc[test_index],
                y_train=y_.iloc[train_index],
                y_test=y_.iloc[test_index],
                directory=fold_directory,
                source_ids=source_ids,
            )
            ns += 1

            # Pick metrics
            metrics = (modified_hausdorff, structural_similarity)

            # Compute UQ with metrics
            index = X_.iloc[test_index].index
            bel_uq(
                index=index,
                directory=fold_directory,
                source_ids=wells,
                metrics=metrics,
                delete=True,
            )

            plot_uq(modified_hausdorff, directory=fold_directory)
            plot_uq(structural_similarity, directory=fold_directory)

    else:
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
            directory=Setup.Directories.forecasts_dir,
            source_ids=source_ids,
        )

        # Pick metrics
        metrics = (modified_hausdorff, structural_similarity)

        # Compute UQ with metrics
        bel_uq(
            index=test_idx,
            directory=Setup.Directories.forecasts_dir,
            source_ids=wells,
            metrics=metrics,
        )

        plot_uq(modified_hausdorff)
        plot_uq(structural_similarity)


def plot_uq(metric_function, directory: str = None):
    if directory is None:
        directory = Setup.Directories.forecasts_dir
    wm = np.load(
        jp(directory, f"uq_{metric_function.__name__}.npy")
    )
    colors = Setup.Wells.colors
    mode_histo(colors=colors, wm=wm, an_i=0, directory=directory, fig_name=metric_function.__name__)


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
    run(model=bel, training_idx=training_r, test_idx=test_r, source_ids=wells, kfold=True, n_splits=5)
