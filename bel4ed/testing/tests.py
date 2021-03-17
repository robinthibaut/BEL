from os.path import join as jp

import numpy as np
from sklearn.model_selection import train_test_split

from bel4ed.config import Setup
from bel4ed.design import UncertaintyQuantification
from bel4ed.datasets import i_am_root


def test_posterior():
    """Compare posterior samples with reference default values"""
    training_file = jp(Setup.Directories.test_dir, "roots.dat")
    test_file = jp(Setup.Directories.test_dir, "test_roots.dat")

    training_r, test_r = i_am_root(training_file=training_file, test_file=test_file)

    wells = np.array([1, 2, 3, 4, 5, 6])

    test_base = Setup
    # Switch forecast directory to test directory
    test_dir = jp(test_base.Directories.test_dir, "forecast")
    test_base.Directories.forecasts_dir = test_dir
    test_base.Wells.combination = wells

    uq = UncertaintyQuantification(
        base=test_base,
        seed=123456,
    )
    # 1 - Fit / Transform
    uq.analysis(
        roots_training=training_r,
        roots_obs=test_r,
    )
    test_base.Wells.combination = wells

    post_mean, post_cov = uq.sample_posterior(n_posts=test_base.HyperParameters.n_posts)

    ref_dir = jp(test_base.Directories.ref_dir, "forecast")

    ref_mean = np.load(jp(ref_dir, test_r[0], "123456", "ref_mean.npy"))
    ref_covariance = np.load(jp(ref_dir, test_r[0], "123456", "ref_covariance.npy"))

    msg1 = "The posterior means are different"
    np.testing.assert_allclose(post_mean, ref_mean, atol=1e-6, err_msg=msg1)

    msg2 = "The posterior covariances are different"
    np.testing.assert_allclose(post_cov, ref_covariance, atol=1e-6, err_msg=msg2)
