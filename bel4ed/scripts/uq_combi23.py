#  Copyright (c) 2021. Robin Thibaut, Ghent University
from os.path import join as jp

from skbel.utils import combinator
from sklearn.model_selection import train_test_split

from bel4ed import init_bel
from bel4ed.config import Setup
from bel4ed.datasets import load_dataset
from bel4ed.design import find_extreme, plot_uq
from bel4ed.metrics import modified_hausdorff, structural_similarity

if __name__ == "__main__":
    combis = combinator([1, 2, 3, 4, 5, 6])
    c23 = list(filter(lambda t: 1 < len(t) < 4, combis))
    name = "c23"
    # Initiate BEL model
    bel = init_bel()
    bel.n_posts = Setup.HyperParameters.n_posts
    bel.mode = "mvn"
    bel.X_shape = (6, 200)
    bel.Y_shape = (1, 100, 87)

    # Load datasets
    X, Y = load_dataset()

    random_state = 8872963

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

    args = [
        (bel, X_train, X_test, y_train, y_test, test_directory, c23, tr)
        for tr in test_roots
    ]

    n_cpu = 8
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
    argsuq = [(bel, y_test, tr, test_directory, c23, metrics) for tr in test_roots]
    # Compute UQ with metrics
    n_cpu = 12
    # pool = mp.Pool(n_cpu)
    # pool.map(bel_uq_mp, argsuq)
    # pool.close()
    # pool.join()

    names = ["MHD", "SSIM"]

    # [
    #     plot_uq(
    #         m,
    #         directory=test_directory,
    #         combi=c23,
    #         title=f"{names[ix]} Training/Test {len(X_train)}/{len(X_test)}",
    #         an_i=ix,
    #     )
    #     for ix, m in enumerate(metrics)
    # ]

    root, comb, mxr, mxc = find_extreme(
        index, modified_hausdorff, [(2, 6)], test_directory
    )
