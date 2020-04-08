import os
import shutil
import uuid
import warnings
from os.path import join as jp
from multiprocessing import Process

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import CCA

from bel.toolbox.file_ops import FileOps
from bel.toolbox.data_ops import DataOps
from bel.toolbox.mesh_ops import MeshOps
from bel.toolbox.plots import Plot
from bel.toolbox.pca_ops import PCAOps
from bel.toolbox.posterior_ops import PosteriorOps
from bel.processing.signed_distance import SignedDistance

plt.style.use('dark_background')


do = DataOps()
mo = MeshOps()
po = PosteriorOps()
x_lim, y_lim, grf = [800, 1150], [300, 700], 1
sd = SignedDistance(x_lim=x_lim, y_lim=y_lim, grf=grf)
mp = Plot(x_lim=x_lim, y_lim=y_lim, grf=grf)


def bel(n_training=250, n_test=5, new_dir=None):
    # Directories
    res_dir = jp('..', 'hydro', 'results')  # Results folders of the hydro simulations

    bel_dir = jp('..', 'forecasts')  # Directory in which to load forecasts

    if new_dir is not None:
        with open(jp(bel_dir, new_dir, 'roots.dat')) as f:
            roots = f.read().splitlines()
    else:
        new_dir = str(uuid.uuid4())  # sub-directory for forecasts
        roots = None

    sub_dir = jp(bel_dir, new_dir)
    obj_dir = jp(sub_dir, 'objects')

    fig_dir = jp(sub_dir, 'figures')
    fig_data_dir = jp(fig_dir, 'Data')
    fig_pca_dir = jp(fig_dir, 'PCA')
    fig_cca_dir = jp(fig_dir, 'CCA')
    fig_pred_dir = jp(fig_dir, 'Predictions')

    # Creates directories
    # TODO: create dirmaker decorator function to pass before saving objects instead of code below
    [FileOps.dirmaker(f) for f in [obj_dir, fig_data_dir, fig_pca_dir, fig_cca_dir, fig_pred_dir]]

    n = n_training + n_test  # Total number of simulations to load.
    check = False  # Flag to check for simulations issues
    tc0, pzs, roots_ = FileOps.load_res(res_dir=res_dir, n=n, check=check, roots=roots)
    # Save file roots
    with open(jp(sub_dir, 'roots.dat'), 'w') as f:
        for r in roots_:
            f.write(os.path.basename(r) + '\n')

    # Compute signed distance on pzs. h is the matrix of target feature on which PCA will be performed
    h = np.array([sd.function(pp) for pp in pzs])
    # Plot all WHPP
    mp.whp(h, fig_file=jp(fig_data_dir, 'all_whpa.png'), show=True)

    # Preprocess d in an arbitrary number of time steps.
    tc = do.d_process(tc0=tc0, n_time_steps=250)
    n_wel = len(tc[0])  # Number of injecting wels

    # Plot d
    mp.curves(tc=tc, n_wel=n_wel, sdir=fig_data_dir)
    mp.curves_i(tc=tc, n_wel=n_wel, sdir=fig_data_dir)

    # %%  PCA

    # Choose size of training and prediction set
    n_sim = len(h)  # Number of simulations
    n_obs = n_test  # Number of 'observations' on which the predictions will be made.
    n_training = n_sim - n_obs  # number of synthetic data that will be used for constructing our prediction model

    if n_training != n_sim - n_obs:
        warnings.warn("The size of training set doesn't correspond with user input")

    load = False  # Whether to load already dumped PCA operator
    # PCA on transport curves
    d_pco = PCAOps(name='d', raw_data=tc, directory=obj_dir)
    d_training, d_prediction = d_pco.pca_tp(n_training)  # Split into training and prediction
    d_pc_training, d_pc_prediction = d_pco.pca_transformation(load=load)

    # PCA on signed distance
    h_pco = PCAOps(name='h', raw_data=h, directory=obj_dir)
    h_training, h_prediction = h_pco.pca_tp(n_training)
    h_pc_training, h_pc_prediction = h_pco.pca_transformation(load=load)

    # Explained variance plots
    mp.explained_variance(d_pco.operator, n_comp=40, fig_file=jp(fig_pca_dir, 'd_exvar.png'), show=True)
    mp.explained_variance(h_pco.operator, n_comp=20, fig_file=jp(fig_pca_dir, 'h_exvar.png'), show=True)

    # Scores plots
    mp.pca_scores(d_pc_training, d_pc_prediction, n_comp=20, fig_file=jp(fig_pca_dir, 'd_scores.png'), show=True)
    mp.pca_scores(h_pc_training, h_pc_prediction, n_comp=20, fig_file=jp(fig_pca_dir, 'h_scores.png'), show=True)

    # Choose number of PCA components to keep.
    # Compares true value with inverse transformation from PCA
    ndo = 45  # Number of components for breakthrough curves
    nho = 30  # Number of components for signed distance

    def pca_inverse_compare():
        n_compare = np.random.randint(n_training)  # Sample number to perform inverse transform comparison
        mp.d_pca_inverse_plot(d_training, n_compare, d_pco.operator, ndo)
        mp.h_pca_inverse_plot(h_training, n_compare, h_pco.operator, nho)
        # Displays the explained variance percentage given the number of components
        print(d_pco.perc_pca_components(ndo))
        print(h_pco.perc_pca_components(nho))

    # Assign final n_comp for PCA
    n_d_pc_comp = ndo
    n_h_pc_comp = nho

    # Cut desired number of PC components
    d_pc_training, d_pc_prediction = d_pco.pca_refresh(n_d_pc_comp)
    h_pc_training, h_pc_prediction = h_pco.pca_refresh(n_h_pc_comp)

    # Save the d and h PC objects.
    joblib.dump(d_pco, jp(obj_dir, 'd_pca.pkl'))
    joblib.dump(h_pco, jp(obj_dir, 'h_pca.pkl'))
    # %% CCA

    n_comp_cca = min(n_d_pc_comp, n_h_pc_comp)  # Number of CCA components is chosen as the min number of PC
    # components between d and h.
    cca = CCA(n_components=n_comp_cca, scale=True, max_iter=int(500 * 2))  # By default, it scales the data
    cca.fit(d_pc_training, h_pc_training)  # Fit
    joblib.dump(cca, jp(obj_dir, 'cca.pkl'))  # Save the fitted CCA operator

    shutil.copy(__file__, jp(fig_dir, 'copied_script.py'))


if __name__ == "__main__":
    multi = 0

    if multi:
        jobs = []
        n_jobs = 4
        for i in range(n_jobs):  # Can run max 4 instances of mt3dms at once on this computer
            process = Process(target=bel)
            jobs.append(process)
            process.start()
        process.join()
        process.close()
    else:
        bel(new_dir='7a362886-38fd-4808-af55-3ceaab752d84')

