from os.path import join as jp

import unittest
import joblib
import warnings

import numpy as np

from experiment.config import Setup
from experiment.design.forecast_error import analysis
from experiment.utils import get_roots


class TestUQ(unittest.TestCase):

    def test_posterior(self):
        """Compare posterior samples with reference"""
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

        ref_post = joblib.load(jp(ref_dir, test_r[0], "123456", "obj", "post.pkl"))
        test_post = joblib.load(jp(test_dir, test_r[0], "123456", "obj", "post.pkl"))

        np.testing.assert_array_equal(test_post.posterior_mean,
                                      ref_post.posterior_mean)

        np.testing.assert_array_equal(test_post.posterior_covariance,
                                      ref_post.posterior_covariance)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        unittest.main()
