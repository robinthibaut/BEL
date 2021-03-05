import unittest
from os.path import join as jp

import joblib
import numpy as np

from experiment.config import Setup
from experiment.learning.bel_pipeline import analysis
from experiment.utils import get_roots


class TestUQ(unittest.TestCase):
    def test_posterior(self):
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

        analysis(
            base=test_base,
            comb=wells,
            roots_training=training_r,
            roots_obs=test_r,
            wipe=False,
            flag_base=True,
        )

        ref_dir = jp(test_base.Directories.ref_dir, "forecast")

        ref_post = joblib.load(
            jp(ref_dir, test_r[0], "123456", "obj", "post.pkl"))
        test_post = joblib.load(
            jp(test_dir, test_r[0], "123456", "obj", "post.pkl"))

        msg1 = "The posterior means are different"
        np.testing.assert_array_equal(test_post.posterior_mean,
                                      ref_post.posterior_mean,
                                      err_msg=msg1)

        msg2 = "The posterior covariances are different"
        np.testing.assert_array_equal(test_post.posterior_covariance,
                                      ref_post.posterior_covariance,
                                      err_msg=msg2)


if __name__ == '__main__':
    unittest.main()
