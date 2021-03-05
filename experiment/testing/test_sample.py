import os
import unittest

from experiment.config import Setup
from experiment.design.forecast_error import analysis
from experiment.utils import get_roots


class TestUQ(unittest.TestCase):
    def test_uq(self):
        training_file = os.path.join(Setup.Directories.test_dir, "roots.dat")
        test_file = os.path.join(Setup.Directories.test_dir, "test_roots.dat")

        training_r, test_r = get_roots(training_file=training_file,
                                       test_file=test_file)

        wells = [[1, 2, 3, 4, 5, 6]]

        test_base = Setup
        # Switch forecast directory to test directory
        test_base.Directories.forecasts_dir = \
            os.path.join(test_base.Directories.test_dir, "forecast")

        analysis(
            base=test_base,
            comb=wells,
            roots_training=training_r,
            roots_obs=test_r,
            wipe=False,
            flag_base=True,
        )

        ref_dir = os.path.join(test_base.Directories.ref_dir, "forecast")

        assert 1 == 0, "OK"


if __name__ == '__main__':
    unittest.main()
