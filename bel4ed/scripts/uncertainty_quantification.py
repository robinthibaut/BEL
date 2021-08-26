#  Copyright (c) 2021. Robin Thibaut, Ghent University
from os.path import join as jp

import numpy as np
import numpy.random
import pandas as pd
from loguru import logger
from sklearn.model_selection import KFold, train_test_split

from bel4ed import kernel_bel, init_bel
from bel4ed.config import Setup
from bel4ed.datasets import i_am_root, load_dataset
from bel4ed.design import bel_training, bel_uq
from bel4ed.goggles import mode_histo
from bel4ed.metrics import modified_hausdorff, structural_similarity


def run(
    model,
    *,
    training_idx: list = None,
    test_idx: list = None,
    source_ids_training: list or np.array = None,
    source_ids_uq: list or np.array = None,
    kfold: bool = False,
    n_splits: int = None,
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
            # logger.info(train_index)
            # logger.info(test_index)
            fold_directory = jp(
                Setup.Directories.forecasts_dir, f"{name}fold_{len(idx)}_split{ns}"
            )

            X_train = X_.iloc[train_index]
            X_test = X_.iloc[test_index]
            y_train = y_.iloc[train_index]
            y_test = y_.iloc[test_index]

            index = X_test.index

            # bel_training(
            #     bel=model,
            #     X_train=X_train,
            #     X_test=X_test,
            #     y_train=y_train,
            #     y_test=y_test,
            #     directory=fold_directory,
            #     source_ids=source_ids_training,
            # )
            ns += 1

            # Pick metrics
            metrics = (modified_hausdorff,)
            acro = ["MHD"]

            # Compute UQ with metrics
            # bel_uq(
            #     bel=model,
            #     index=index,
            #     directory=fold_directory,
            #     source_ids=source_ids_uq,
            #     metrics=metrics,
            #     delete=True,
            #     clear=True,
            # )

            [
                plot_uq(
                    m,
                    directory=fold_directory,
                    title=f"{acro[ix]} Training/Test {len(X_train)}/{len(X_test)}",
                    an_i=ix,
                )
                for ix, m in enumerate(metrics)
            ]
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
            source_ids=source_ids_training,
        )

        # Pick metrics
        metrics = (modified_hausdorff, structural_similarity)
        acro = ["MHD", "SSIM"]

        # Compute UQ with metrics
        if len(test_idx) > 1:
            bel_uq(
                bel_mod=model,
                root_index=test_idx,
                directory=custom_directory,
                source_ids=source_ids_uq,
                metrics=metrics,
            )

            [
                plot_uq(
                    m,
                    directory=custom_directory,
                    title=f"{acro[ix]} Training/Test {len(training_idx)}/{len(test_idx)}",
                    an_i=ix + 2,
                )
                for ix, m in enumerate(metrics)
            ]
    else:
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
            bel_mod=model,
            root_index=index,
            directory=test_directory,
            source_ids=source_ids_uq,
            metrics=metrics,
            delete=False,
        )

        # try:
        #     [
        #         plot_uq(
        #             m,
        #             directory=test_directory,
        #             title=f"{m.__name__.capitalize()} Training/Test {len(X_train)}/{len(X_test)}",
        #             an_i=ix,
        #         )
        #         for ix, m in enumerate(metrics)
        #     ]
        # except ValueError:
        #     pass


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
    # Get roots used for testing
    training_file = jp(Setup.Directories.storage_dir, "training_roots_400.dat")
    test_file = jp(Setup.Directories.storage_dir, "test_roots_100.dat")
    training_r, test_r = i_am_root(training_file=training_file, test_file=test_file)
    test_r = ["818bf1676c424f76b83bd777ae588a1d"]
    # %%
    # Source IDs
    wells_uq = np.array([[1], [2], [3], [4], [5], [6]], dtype=object)
    wells_training = np.array([[1], [2], [3], [4], [5], [6]], dtype=object)

    wells_training = np.array([[1, 2, 3, 4, 5, 6]], dtype=object)
    # %%
    # Initiate BEL model
    bel = init_bel()
    bel.n_posts = Setup.HyperParameters.n_posts
    bel.mode = "mvn"
    bel.X_shape = (6, 200)  # Six curves with 200 time steps each
    bel.Y_shape = (1, 100, 87)  # One matrix with 100 rows and 87 columns
    # Train model
    # %%
    # Reference dataset
    run(
        model=bel,
        training_idx=training_r,
        test_idx=test_r,
        source_ids_training=wells_training,
        name="IAH",
        # source_ids_uq=wells_uq,
    )
    # %%
    # Test
    # idx_ = [*training_r, *test_r]
    # training_test = idx_[50:]
    # test_test = idx_[:50]
    # run(model=bel, training_idx=training_test, test_idx=test_test, source_ids=wells, name="check")
    # %%
    # KFold on custom dataset
    # run(
    #     model=bel,
    #     training_idx=training_r,
    #     test_idx=test_r,
    #     source_ids_training=wells_training,
    #     # train_size=1000,
    #     # test_size=250,
    #     kfold=True,
    #     n_splits=5,
    #     shuffle=False,
    #     random_state=None,
    #     name="K_update_",
    # )
    # %%
    # run(
    #     model=bel,
    #     # training_idx=training_r,
    #     # test_idx=test_r,
    #     train_size=200,
    #     test_size=50,
    #     kfold=True,
    #     n_splits=5,
    #     shuffle=False,
    #     random_state=None,
    #     name="k",
    # )
    # %%
    # Test datasets with various sizes
    sizes = np.concatenate([range(125, 500, 25), range(500, 1000, 100)])

    rand = [
        287071,
        437176,
        446122,
        502726,
        656184,
        791302,
        824883,
        851117,
        885166,
        980285,
        15843,
        202157,
        235506,
        430849,
        547976,
        617924,
        862286,
        863668,
        975934,
        993935,
        1561,
        1998,
        1678941,
        19619,
        125691,
        168994652,
        16516,
        5747,
        156886,
        218766,
        21518,
        51681,
        6546844,
        5418717,
    ]

    # for rs in rand:
    #     for i in sizes:
    #         bel.n_posts = 400
    #         run(
    #             model=bel,
    #             source_ids_training=wells_training,
    #             train_size=i,
    #             test_size=1,
    #             shuffle=True,
    #             random_state=rs,
    #             name="n_thr",
    #         )
    # %%
    # run(model=bel,
    #     source_ids_training=wells_training,
    #     train_size=400,
    #     test_size=100,
    #     shuffle=True)
