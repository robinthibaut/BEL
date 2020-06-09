#  Copyright (c) 2020. Robin Thibaut, Ghent University

"""
Forecast error analysis
"""

import os
from os.path import join as jp

import joblib
import matplotlib.pyplot as plt
import numpy as np
import vtk
from sklearn.neighbors import KernelDensity

from experiment.base.inventory import Directories, Focus, Wels
from experiment.goggles.visualization import Plot, cca_plot
from experiment.math.hausdorff import modified_distance
from experiment.math.postio import PosteriorIO
from experiment.math.signed_distance import SignedDistance
from experiment.toolbox import filesio as fops

plt.style.use('dark_background')


class UncertaintyQuantification:

    def __init__(self, study_folder, wel_comb=None):
        """

        :param study_folder: Name of the folder in the 'forecast' directory on which UQ will be performed.
        """
        self.po = PosteriorIO()
        fc = Focus()
        self.x_lim, self.y_lim, self.grf = fc.x_range, fc.y_range, fc.cell_dim

        self.wel_comb = wel_comb
        self.mplot = Plot(x_lim=self.x_lim, y_lim=self.y_lim, grf=self.grf, wel_comb=self.wel_comb)

        # Directories & files paths
        md = Directories()
        self.main_dir = md.main_dir

        self.grid_dir = md.grid_dir
        self.mplot.wdir = self.grid_dir

        # TODO: get folders from base model
        self.bel_dir = jp(self.main_dir, 'bel', 'forecasts', study_folder)
        self.base_dir = jp(os.path.dirname(self.bel_dir), 'base', 'obj')
        self.res_dir = jp(self.bel_dir, 'obj')
        self.fig_cca_dir = jp(self.bel_dir, 'cca')
        self.fig_pred_dir = jp(self.bel_dir, 'uq')

        # Load objects
        f_names = list(map(lambda fn: jp(self.res_dir, fn + '.pkl'), ['cca', 'd_pca']))
        self.cca_operator, self.d_pco = list(map(joblib.load, f_names))
        self.h_pco = joblib.load(jp(self.base_dir, 'h_pca.pkl'))

        # Inspect transformation between physical and PC space
        dnc0 = self.d_pco.ncomp
        hnc0 = self.h_pco.ncomp
        # print(d_pco.perc_pca_components(dnc0))
        # print(h_pco.perc_pca_components(hnc0))
        self.mplot.pca_inverse_compare(self.d_pco, self.h_pco, dnc0, hnc0)

        # Cut desired number of PC components
        d_pc_training, d_pc_prediction = self.d_pco.pca_refresh(dnc0)
        h_pc_training, h_pc_prediction = self.h_pco.pca_refresh(hnc0)

        # CCA plots
        d_cca_training, h_cca_training = self.cca_operator.transform(d_pc_training, h_pc_training)
        d_cca_training, h_cca_training = d_cca_training.T, h_cca_training.T

        cca_plot(self.cca_operator, d_cca_training, h_cca_training, d_pc_prediction, h_pc_prediction,
                 sdir=self.fig_cca_dir)

        # Sampling
        self.n_training = len(d_pc_training)
        self.n_test = len(d_pc_prediction)
        self.sample_n = 0
        self.n_posts = 500
        self.forecast_posterior = None
        self.d_pc_obs = None
        self.h_true_obs = None
        self.shape = None
        self.h_pc_true_pred = None
        self.h_pred = None

        # Contours
        self.vertices = None

    # %% Random sample from the posterior
    def sample_posterior(self, sample_n=None, n_posts=None):
        """
        Extracts n random samples from the posterior
        :param sample_n: Sample identifier
        :param n_posts: Desired number of samples
        :return:
        """
        if sample_n is not None:
            self.sample_n = sample_n
        if n_posts is not None:
            self.n_posts = n_posts

        self.forecast_posterior = self.po.random_sample(sample_n=self.sample_n,
                                                        pca_d=self.d_pco,
                                                        pca_h=self.h_pco,
                                                        cca_obj=self.cca_operator,
                                                        n_posts=self.n_posts,
                                                        add_comp=0)
        # Get the true array of the prediction
        # Prediction set - PCA space
        self.d_pc_obs = self.d_pco.predict_pc[sample_n]
        self.shape = self.h_pco.raw_data.shape
        # Prediction set - physical space
        self.h_true_obs = self.h_pco.predict_physical[sample_n].reshape(self.shape[1], self.shape[2])

        # Predicting the function based for a certain number of 'observations'
        self.h_pc_true_pred = self.cca_operator.predict(self.d_pc_obs[:self.d_pco.ncomp].reshape(1, -1))
        # Going back to the original function dimension and reshape.
        self.h_pred = self.h_pco.inverse_transform(self.h_pc_true_pred).reshape(self.shape[1], self.shape[2])

        # Plot results
        ff = jp(self.fig_pred_dir, '{}_{}.pdf'.format(sample_n, self.cca_operator.n_components))
        self.mplot.whp_prediction(forecasts=self.forecast_posterior,
                                  h_true=self.h_true_obs,
                                  h_pred=self.h_pred,
                                  show_wells=True,
                                  fig_file=ff)

    # %% extract 0 contours
    def c0(self, write_vtk=1):
        """
        Extract the 0 contour from the sampled posterior, corresponding to the WHPA delineation
        :param write_vtk: Boolean flag to export VTK files
        """
        self.vertices = self.mplot.contours_vertices(self.forecast_posterior)
        if write_vtk:
            vdir = jp(self.fig_pred_dir, '{}_vtk'.format(self.sample_n))
            fops.dirmaker(vdir)
            for i, v in enumerate(self.vertices):
                nv = len(v)
                points = vtk.vtkPoints()
                [points.InsertNextPoint(np.insert(c, 2, 0)) for c in v]
                # Create a polydata to store everything in
                polyData = vtk.vtkPolyData()
                # Add the points to the dataset
                polyData.SetPoints(points)
                # Create a cell array to store the lines in and add the lines to it
                cells = vtk.vtkCellArray()
                cells.InsertNextCell(nv)
                [cells.InsertCellPoint(k) for k in range(nv)]
                # Add the lines to the dataset
                polyData.SetLines(cells)
                # Export
                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetInputData(polyData)

                writer.SetFileName(jp(vdir, 'forecast_posterior_{}.vtp'.format(i)))
                writer.Write()

    # %% Kernel density
    def kernel_density(self):
        # Scatter plot vertices
        # nn = sample_n
        # plt.plot(vertices[nn][:, 0], vertices[nn][:, 1], 'o-')
        # plt.show()

        # Grid geometry
        xmin = self.x_lim[0]
        xmax = self.x_lim[1]
        ymin = self.y_lim[0]
        ymax = self.y_lim[1]
        # Create a structured grid to estimate kernel density
        # TODO: create a function to copy/paste values on differently refined grids
        # Prepare the Plot instance with right dimensions
        grf_kd = 4
        mpkde = Plot(x_lim=self.x_lim, y_lim=self.y_lim, grf=grf_kd)
        mpkde.wdir = self.grid_dir
        cell_dim = grf_kd
        xgrid = np.arange(xmin, xmax, cell_dim)
        ygrid = np.arange(ymin, ymax, cell_dim)
        X, Y = np.meshgrid(xgrid, ygrid)
        # x, y coordinates of the grid cells vertices
        xy = np.vstack([X.ravel(), Y.ravel()]).T

        # Define a disk within which the KDE will be performed to save time
        x0, y0, radius = 1000, 500, 200
        r = np.sqrt((xy[:, 0] - x0) ** 2 + (xy[:, 1] - y0) ** 2)
        inside = r < radius
        xyu = xy[inside]  # Create mask

        # Perform KDE
        bw = 1.  # Arbitrary 'smoothing' parameter
        # Reshape coordinates
        x_stack = np.hstack([vi[:, 0] for vi in self.vertices])
        y_stack = np.hstack([vi[:, 1] for vi in self.vertices])
        # Final array np.array([[x0, y0],...[xn,yn]])
        xykde = np.vstack([x_stack, y_stack]).T
        kde = KernelDensity(kernel='gaussian',  # Fit kernel density
                            bandwidth=bw).fit(xykde)
        score = np.exp(kde.score_samples(xyu))  # Sample at the desired grid cells

        def score_norm(sc, max_score=None):
            """
            Normalizes the KDE scores.
            """
            sc -= sc.min()
            sc /= sc.max()

            sc += 1
            sc = sc ** -1

            sc -= sc.min()
            sc /= sc.max()

            return sc

        # Normalize
        score = score_norm(score)

        # Assign the computed scores to the grid
        z = np.full(inside.shape, 1, dtype=float)  # Create array filled with 1
        z[inside] = score
        z = np.flipud(z.reshape(X.shape))  # Flip to correspond to actual distribution.

        # Plot KDE
        self.mplot.whp(self.h_true_obs.reshape(1, self.shape[1], self.shape[2]),
                       alpha=1,
                       lw=1,
                       show_wells=True,
                       colors='red',
                       show=False)
        mpkde.whp(bkg_field_array=z,
                  vmin=None,
                  vmax=None,
                  cmap='RdGy',
                  colors='red',
                  fig_file=jp(self.fig_pred_dir, '{}comp.pdf'.format(self.sample_n)),
                  show=True)

        return z

    # %% New approach : stack binary WHPA
    def binary_stack(self):
        """
        Takes WHPA vertices and binarizes the image (e.g. 1 inside, 0 outside WHPA).
        """
        # For this approach we use our SignedDistance module
        sd_kd = SignedDistance(x_lim=self.x_lim, y_lim=self.y_lim, grf=2)  # Initiate SD object
        mpbin = Plot(x_lim=self.x_lim, y_lim=self.y_lim, grf=2, wel_comb=self.wel_comb)  # Initiate Plot tool
        mpbin.wdir = self.grid_dir
        # Create binary images of WHPA stored in bin_whpa
        bin_whpa = [sd_kd.matrix_poly_bin(pzs=p, inside=1 / self.n_posts, outside=0) for p in self.vertices]
        big_sum = np.sum(bin_whpa, axis=0)  # Stack them
        b_low = np.where(big_sum == 0, 1, big_sum)  # Replace 0 values by 1
        b_low = np.flipud(b_low)

        # a measure of the error could be a measure of the area covered by the n samples.
        error_estimate = len(np.where(b_low < 1)[0])  # Number of cells covered at least once.

        # Display result
        self.mplot.whp(self.h_true_obs.reshape(1, self.shape[1], self.shape[2]),
                       alpha=1,
                       lw=1,
                       show_wells=False,
                       colors='red',
                       show=False)

        mpbin.whp(bkg_field_array=b_low,
                  show_wells=True,
                  vmin=None,
                  vmax=None,
                  cmap='RdGy',
                  fig_file=jp(self.fig_pred_dir, '{}_0stacked.pdf'.format(self.sample_n)),
                  title=str(error_estimate),
                  show=True)

        return error_estimate

    #  Let's try Hausdorff...
    def mhd(self):
        """Computes the modified Hausdorff distance"""

        # Delineation vertices of the true array
        v_h_true = self.mplot.contours_vertices(self.h_true_obs)[0]

        # Compute MHD between the true vertices and the n sampled vertices
        mhds = np.array([modified_distance(v_h_true, vt) for vt in self.vertices])

        # Identify the closest and farthest results
        min_pos = np.where(mhds == np.min(mhds))[0][0]
        max_pos = np.where(mhds == np.max(mhds))[0][0]

        # Plot results
        fig = jp(self.fig_pred_dir, '{}_{}_hausdorff.pdf'.format(self.sample_n, self.cca_operator.n_components))
        self.mplot.whp_prediction(forecasts=np.expand_dims(self.forecast_posterior[max_pos], axis=0),
                                  h_true=self.h_true_obs,
                                  h_pred=self.forecast_posterior[min_pos],
                                  show_wells=True,
                                  title=str(np.round(mhds.mean(), 2)),
                                  fig_file=fig)

        return mhds
