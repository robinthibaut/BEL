#  Copyright (c) 2020. Robin Thibaut, Ghent University

from experiment.bel.forecast_error import UncertaintyQuantification

if __name__ == '__main__':
    # sf = dcp.bel(n_test=10)
    uq = UncertaintyQuantification(study_folder='5d78b303-45b4-4dce-8fb4-3ef0cd14bf1c')
    for i in range(uq.n_test):
        uq.sample_posterior(sample_n=i, n_posts=500)
        uq.c0(write_vtk=0)
        uq.mhd()
