from os.path import join as jp

import numpy as np

from bel4ed.config import Setup
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.pipeline import Pipeline

from bel4ed.datasets import i_am_root, load_dataset
from bel4ed.learning.bel import BEL


def test_posterior():
    """Compare posterior samples with reference default values"""

    # Get roots used for testing
    training_file = jp(Setup.Directories.test_dir, "roots.dat")
    test_file = jp(Setup.Directories.test_dir, "test_roots.dat")
    training_r, test_r = i_am_root(training_file=training_file, test_file=test_file)

    # Load datasets
    X, Y = load_dataset()

    # Source IDs
    wells = np.array([1, 2, 3, 4, 5, 6])

    # Select roots for testing
    X_train = X.loc[training_r]
    X_test = X.loc[test_r]
    y_train = Y.loc[training_r]

    # Modify config file
    test_base = Setup
    # Switch forecast directory to test directory
    test_dir = jp(test_base.Directories.test_dir, "forecast")
    test_base.Directories.forecasts_dir = test_dir
    test_base.Wells.combination = wells

    # Set seed
    seed = 123456
    np.random.seed(seed)

    # Number of CCA components is chosen as the min number of PC
    n_pc_pred, n_pc_targ = (
        Setup.HyperParameters.n_pc_predictor,
        Setup.HyperParameters.n_pc_target,
    )

    # Pipeline before CCA
    X_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("pca", PCA()),
        ]
    )
    Y_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("pca", PCA()),
        ]
    )

    # Canonical Correlation Analysis
    cca = CCA(n_components=min(n_pc_targ, n_pc_pred), max_iter=500 * 20, tol=1e-6)

    # Pipeline after CCA
    X_post_processing = Pipeline(
        [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
    )
    Y_post_processing = Pipeline(
        [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
    )

    # Initiate BEL object
    bel = BEL(
        X_pre_processing=X_pre_processing,
        X_post_processing=X_post_processing,
        Y_pre_processing=Y_pre_processing,
        Y_post_processing=Y_post_processing,
        cca=cca,
    )

    # Fit
    bel.fit(X=X_train, Y=y_train)
    # Predict posterior mean and covariance
    post_mean, post_cov = bel.predict(X_test)

    # Compare with reference
    ref_dir = jp(test_base.Directories.ref_dir, "forecast")

    ref_mean = np.load(jp(ref_dir, test_r[0], "123456", "ref_mean.npy"))
    ref_covariance = np.load(jp(ref_dir, test_r[0], "123456", "ref_covariance.npy"))

    msg1 = "The posterior means are different"
    np.testing.assert_allclose(post_mean, ref_mean, atol=1e-6, err_msg=msg1)

    msg2 = "The posterior covariances are different"
    np.testing.assert_allclose(post_cov, ref_covariance, atol=1e-6, err_msg=msg2)
