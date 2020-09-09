#  Copyright (c) 2020. Robin Thibaut, Ghent University

"""
Forecast error analysis.

- quantifying uncertainty
- assessing uncertainty
- modeling uncertainty
- realistic assessment of uncertainty.

- Jef Caers, Modeling Uncertainty in the Earth Sciences, p. 50
"""

import os
from os.path import join as jp

import joblib
import matplotlib.pyplot as plt
import numpy as np
import vtk
from sklearn.neighbors import KernelDensity

from experiment.goggles.visualization import Plot
from experiment.math.hausdorff import modified_distance
from experiment.math.postio import PosteriorIO
from experiment.math.signed_distance import SignedDistance
from experiment.toolbox import filesio as fops

plt.style.use('dark_background')


class UncertaintyQuantification:

    def __init__(self,
                 base,
                 study_folder,
                 base_dir=None,
                 wel_comb=None,
                 seed=None):
        """

        :param base: class: Base object (inventory)
        :param study_folder: str: Name of the root uuid in the 'forecast' directory on which UQ will be performed
        :param base_dir: str: Path to base directory
        :param wel_comb: list: List of data source combinations
        :param seed: int: Seed
        """

        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None

        self.base = base

        fc = self.base.Focus()
        self.x_lim, self.y_lim, self.grf = fc.x_range, fc.y_range, fc.cell_dim

        self.wel_comb = wel_comb
        self.mplot = Plot(x_lim=self.x_lim, y_lim=self.y_lim, grf=self.grf, wel_comb=self.wel_comb)

        # Directories & files paths
        md = self.base.Directories()
        self.main_dir = md.main_dir

        self.grid_dir = md.grid_dir
        self.mplot.wdir = self.grid_dir

        # TODO: get folders from base model
        self.bel_dir = jp(md.forecasts_dir, study_folder)
        if base_dir is None:
            self.base_dir = jp(os.path.dirname(self.bel_dir), 'base', 'obj')
        else:
            self.base_dir = base_dir
        self.res_dir = jp(self.bel_dir, 'obj')
        self.fig_cca_dir = jp(self.bel_dir, 'cca')
        self.fig_pred_dir = jp(self.bel_dir, 'uq')

        self.po = PosteriorIO(directory=self.res_dir)

        # Load objects
        f_names = list(map(lambda fn: jp(self.res_dir, fn + '.pkl'), ['cca', 'd_pca']))
        self.cca_operator, self.d_pco = list(map(joblib.load, f_names))
        self.h_pco = joblib.load(jp(self.base_dir, 'h_pca.pkl'))

        # Inspect transformation between physical and PC space
        dnc0 = self.d_pco.ncomp
        hnc0 = self.h_pco.ncomp

        # Cut desired number of PC components
        d_pc_training, self.d_pc_prediction = self.d_pco.pca_refresh(dnc0)
        self.h_pco.pca_refresh(hnc0)

        # Sampling
        self.n_training = len(d_pc_training)
        self.sample_n = 0  # This class used to take into account multiple observations, now this parameter remains
        # fixed to 0.
        self.n_posts = self.base.Forecast.n_posts
        self.forecast_posterior = None
        self.h_true_obs = None  # True h in physical space
        self.shape = None
        self.h_pc_true_pred = None  # CCA predicted 'true' h PC
        self.h_pred = None  # 'true' h in physical space

        # Contours
        self.vertices = None

    # %% Random sample from the posterior
    def sample_posterior(self, sample_n=None, n_posts=None, save_target_pc=True):
        """
        Extracts n random samples from the posterior
        :param sample_n: int: Sample identifier
        :param n_posts: int: Desired number of samples
        :param save_target_pc: bool: Flag to save the observation target PC
        :return:
        """
        if sample_n is not None:
            self.sample_n = sample_n

        if n_posts is not None:
            self.n_posts = n_posts

        # Extract n random sample (target pc's).
        # The posterior distribution is computed within the method below.
        self.forecast_posterior = self.po.random_sample(pca_d=self.d_pco,
                                                        pca_h=self.h_pco,
                                                        cca_obj=self.cca_operator,
                                                        n_posts=self.n_posts,
                                                        add_comp=False)
        # if save_target_pc:
        #     fname = jp(self.res_dir, f'{self.n_posts}_target_pc.npy')
        #     np.save(fname, forecast_pc)

        # Generate forecast in the initial dimension and reshape.
        # self.forecast_posterior = \
        #     self.h_pco.inverse_transform(forecast_pc).reshape((n_posts,
        #                                                        self.h_pco.training_shape[1],
        #                                                        self.h_pco.training_shape[2]))

        # np.save(jp(self.res_dir, 'forecast_posterior.npy'), self.forecast_posterior)

        # Get the true array of the prediction
        # Prediction set - PCA space
        self.shape = self.h_pco.training_shape
        # Prediction set - physical space
        self.h_true_obs = self.h_pco.predict_physical[sample_n].reshape(self.shape[1], self.shape[2])

        np.save(jp(self.res_dir, 'h_true_obs.npy'), self.h_true_obs)

        # Predicting the function based for a certain number of 'observations'
        self.h_pc_true_pred = self.cca_operator.predict(self.d_pc_prediction)

        # Going back to the original function dimension and reshape.
        self.h_pred = self.h_pco.inverse_transform(self.h_pc_true_pred).reshape(self.shape[1], self.shape[2])

        np.save(jp(self.res_dir, 'h_pred.npy'), self.h_pred)

    # %% extract 0 contours
    def c0(self, write_vtk=1):
        """
        Extract the 0 contour from the sampled posterior, corresponding to the WHPA delineation
        :param write_vtk: bool: Flag to export VTK files
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
                poly_data = vtk.vtkPolyData()
                # Add the points to the dataset
                poly_data.SetPoints(points)
                # Create a cell array to store the lines in and add the lines to it
                cells = vtk.vtkCellArray()
                cells.InsertNextCell(nv)
                [cells.InsertCellPoint(k) for k in range(nv)]
                # Add the lines to the dataset
                poly_data.SetLines(cells)
                # Export
                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetInputData(poly_data)

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
        # self.mplot.whp(self.h_true_obs.reshape(1, self.shape[1], self.shape[2]),
        #                alpha=1,
        #                lw=1,
        #                show_wells=True,
        #                colors='red',
        #                show=False)
        # mpkde.whp(bkg_field_array=z,
        #           vmin=None,
        #           vmax=None,
        #           cmap='RdGy',
        #           colors='red',
        #           fig_file=jp(self.fig_pred_dir, '{}comp.png'.format(self.sample_n)),
        #           show=True)

        return z

    # %% New approach : stack binary WHPA
    def binary_stack(self):
        """
        Takes WHPA vertices and binarizes the image (e.g. 1 inside, 0 outside WHPA).
        """
        # For this approach we use our SignedDistance module
        sd_kd = SignedDistance(x_lim=self.x_lim, y_lim=self.y_lim, grf=4)  # Initiate SD object
        mpbin = Plot(x_lim=self.x_lim, y_lim=self.y_lim, grf=4, wel_comb=self.wel_comb)  # Initiate Plot tool
        mpbin.wdir = self.grid_dir
        # Create binary images of WHPA stored in bin_whpa
        bin_whpa = [sd_kd.matrix_poly_bin(pzs=p, inside=1 / self.n_posts, outside=0) for p in self.vertices]
        big_sum = np.sum(bin_whpa, axis=0)  # Stack them
        b_low = np.where(big_sum == 0, 1, big_sum)  # Replace 0 values by 1
        b_low = np.flipud(b_low)

        # a measure of the error could be a measure of the area covered by the n samples.
        # error_estimate = len(np.where(b_low < 1)[0])  # Number of cells covered at least once.

        # Display result
        # self.mplot.whp(self.h_true_obs.reshape(1, self.shape[1], self.shape[2]),
        #                alpha=1,
        #                lw=1,
        #                show_wells=False,
        #                colors='red',
        #                show=False)
        #
        # mpbin.whp(bkg_field_array=b_low,
        #           show_wells=True,
        #           vmin=None,
        #           vmax=None,
        #           cmap='RdGy',
        #           fig_file=jp(self.fig_pred_dir, '{}_0stacked.png'.format(self.sample_n)),
        #           title=str(error_estimate),
        #           show=True)

        # Save result
        np.save(jp(self.res_dir, 'bin'), b_low)

    #  Let's try Hausdorff...
    def mhd(self):
        """
        Computes the Modified Hausdorff Distance between the true WHPA that has been recovered from its n first PCA
        components to allow proper comparison.
        """

        # The new idea is to compute MHD with the observed WHPA recovered from it's n first PC.
        n_cut = self.h_pco.ncomp  # Number of components to keep
        # Inverse transform and reshape
        v_h_true_cut = \
            self.h_pco.inverse_transform(self.h_pco.predict_pc, n_cut).reshape((self.shape[1], self.shape[2]))

        # Delineation vertices of the true array
        v_h_true = self.mplot.contours_vertices(v_h_true_cut)[0]

        # Compute MHD between the true vertices and the n sampled vertices
        mhds = np.array([modified_distance(v_h_true, vt) for vt in self.vertices])

        # Identify the closest and farthest results
        # min_pos = np.where(mhds == np.min(mhds))[0][0]
        # max_pos = np.where(mhds == np.max(mhds))[0][0]

        # Plot results
        # fig = jp(self.fig_pred_dir, '{}_{}_hausdorff.png'.format(self.sample_n, self.cca_operator.n_components))
        # self.mplot.whp_prediction(  # forecasts=np.expand_dims(self.forecast_posterior[max_pos], axis=0),
        #     forecasts=None,
        #     h_true=v_h_true_cut,
        #     h_pred=self.forecast_posterior[min_pos],
        #     show_wells=True,
        #     title=str(np.round(mhds.mean(), 2)),
        #     fig_file=fig)

        # Save mhd
        np.save(jp(self.res_dir, 'haus'), mhds)
