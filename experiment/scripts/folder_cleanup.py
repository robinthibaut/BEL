#  Copyright (c) 2021. Robin Thibaut, Ghent University

from experiment.utils import cleanup, filter_file

if __name__ == "__main__":
    cleanup()
    filter_file("pz.npy")
    # spare_me()
