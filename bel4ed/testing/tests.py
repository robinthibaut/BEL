from os.path import join as jp

import joblib
import numpy as np

from bel4ed import design
from bel4ed.config import Setup
from bel4ed.utils import get_roots

from bel4ed.algorithms import modified_hausdorff, structural_similarity


def test_posterior():
    """Compare posterior samples with reference default values"""
    training_file = jp(Setup.Directories.test_dir, "roots.dat")
    test_file = jp(Setup.Directories.test_dir, "test_roots.dat")

    training_r, test_r = get_roots(training_file=training_file,
                                   test_file=test_file)

    wells = [[1, 2, 3, 4, 5, 6]]

    test_base = Setup
    # Switch forecast directory to test directory
    test_dir = jp(test_base.Directories.test_dir, "forecast")
    test_base.Directories.forecasts_dir = test_dir
    test_base.Wells.combination = wells
    test_base.ED.metric = structural_similarity

    design.analysis(
        base=test_base,
        roots_training=training_r,
        roots_obs=test_r,
        wipe=False,
        flag_base=True,
    )

    ref_dir = jp(test_base.Directories.ref_dir, "forecast")

    ref_mean = np.load(jp(ref_dir, test_r[0], "123456", "ref_mean.npy"))
    ref_covariance = np.load(jp(ref_dir, test_r[0], "123456", "ref_covariance.npy"))

    test_post = joblib.load(jp(test_dir, test_r[0], "123456", "obj", "post.pkl"))

    msg1 = "The posterior means are different"
    np.testing.assert_allclose(test_post.posterior_mean,
                               ref_mean,
                               atol=1e-6,
                               err_msg=msg1)

    msg2 = "The posterior covariances are different"
    np.testing.assert_allclose(test_post.posterior_covariance,
                               ref_covariance,
                               atol=1e-6,
                               err_msg=msg2)
