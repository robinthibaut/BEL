#  Copyright (c) 2021. Robin Thibaut, Ghent University
from bel4ed import Setup
from bel4ed.datasets._base import (
    cleanup,
    filter_file,
    remove_file,
    spare_me,
    count_file,
)

if __name__ == "__main__":
    cleanup()
    # filter_file("pz.npy")
    n = count_file(res_tree=Setup.Directories.hydro_res_dir, file_name="bkt.npy")
    print(n)
    # spare_me()
    # n = count_file(res_tree=Setup.Directories.hydro_res_dir, file_name="bkt.npy")
    # print(n)
    # remove_file(
    #     "/Users/robin/PycharmProjects/BEL/bel4ed/datasets/fwd_structural", "hk0.npy"
    # )
