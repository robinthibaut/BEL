#  Copyright (c) 2020. Robin Thibaut, Ghent University

from experiment.bel.forecast_error import UncertaintyQuantification
from experiment.processing import decomposition as dcp

if __name__ == '__main__':
    dcp.bel(n_test=10)
    uq = UncertaintyQuantification(study_folder='03f189e4-a7dc-4a09-b1d6-294ff27e7aed')
    for i in range(uq.n_test):
        uq.sample_posterior(sample_n=i, n_posts=500)
        uq.c0(write_vtk=0)
        uq.mhd()
