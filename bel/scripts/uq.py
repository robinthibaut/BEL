#  Copyright (c) 2020. Robin Thibaut, Ghent University

from bel.bel.forecast_error import UncertaintyQuantification

if __name__ == '__main__':
    # dc.bel(n_test=20)
    uq = UncertaintyQuantification(study_folder='1e77f10f-6bbb-4521-abfb-14be26699f94')
    for i in range(uq.n_test):
        uq.sample_posterior(sample_n=i, n_posts=500)
        uq.c0(write_vtk=0)
        uq.mhd()
