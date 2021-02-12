#  Copyright (c) 2021. Robin Thibaut, Ghent University

"""
Forecast error analysis.

- quantifying uncertainty
- assessing uncertainty
- modeling uncertainty
- realistic assessment of uncertainty

- Jef Caers, Modeling Uncertainty in the Earth Sciences, p. 50
"""

import os
from os.path import join as jp

import joblib
import numpy as np
import vtk
from sklearn.neighbors import KernelDensity

from experiment.calculation.postio import PosteriorIO
from experiment.spatial.distance import grid_parameters, modified_hausdorff
from experiment.spatial.grid import binary_polygon, contours_vertices, refine_machine
from experiment.toolbox import filesio as fops


class UncertaintyQuantification:

    def __init__(self,
                 base,
                 study_folder: str,
                 base_dir: str = None,
                 wel_comb: list = None,
                 seed: int = None):
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

        fc = self.base.focus()
        self.x_lim, self.y_lim, self.grf = fc.x_range, fc.y_range, fc.cell_dim

        self.wel_comb = wel_comb

        # Directories & files paths
        md = self.base.directories()
        self.main_dir = md.main_dir

        self.grid_dir = md.grid_dir

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
        dnc0 = self.d_pco.n_pc_cut
        hnc0 = self.h_pco.n_pc_cut

        # Cut desired number of PC components
        d_pc_training, self.d_pc_prediction = self.d_pco.pca_refresh(dnc0)
        self.h_pco.pca_refresh(hnc0)

        # Sampling
        self.n_training = len(d_pc_training)
        self.n_posts = self.base.forecast.n_posts
        self.forecast_posterior = None
        self.h_true_obs = None  # True h in physical space
        self.shape = None
        self.h_pc_true_pred = None  # CCA predicted 'true' h PC
        self.h_pred = None  # 'true' h in physical space

        # 0 contours of posterior WHPA
        self.vertices = None

# %% Random sample from the posterior
    def sample_posterior(self,
                         n_posts: int = None):
        """
        Extracts n_posts random samples from the posterior.
        :param n_posts: int: Desired number of samples
        :return:
        """

        if n_posts is not None:
            self.n_posts = n_posts

        # Extract n random sample (target pc's).
        # The posterior distribution is computed within the method below.
        self.forecast_posterior = self.po.bel_predict(pca_d=self.d_pco,
                                                      pca_h=self.h_pco,
                                                      cca_obj=self.cca_operator,
                                                      n_posts=self.n_posts,
                                                      add_comp=False)

        # Get the true array of the prediction
        # Prediction set - PCA space
        self.shape = self.h_pco.training_shape

# %% extract 0 contours
    def c0(self,
           write_vtk: bool = 1):
        """
        Extract the 0 contour from the sampled posterior, corresponding to the WHPA delineation
        :param write_vtk: bool: Flag to export VTK files
        """
        nrow, ncol, x, y = refine_machine(self.x_lim, self.y_lim, self.grf)
        self.vertices = contours_vertices(x, y, self.forecast_posterior)
        if write_vtk:
            vdir = jp(self.fig_pred_dir, 'vtk')
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

                writer.SetFileName(jp(vdir, f'forecast_posterior_{i}.vtp'))
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

        return z

# %% New approach : stack binary WHPA
    def uq_binary_stack(self):
        """
        Takes WHPA vertices and binarizes the image (e.g. 1 inside, 0 outside WHPA).
        """
        xys, nrow, ncol = grid_parameters(x_lim=self.x_lim, y_lim=self.y_lim, grf=self.grf)  # Initiate SD object
        # Create binary images of WHPA stored in bin_whpa
        bin_whpa = [binary_polygon(xys, nrow, ncol, pzs=p, inside=1 / self.n_posts, outside=0) for p in self.vertices]
        big_sum = np.sum(bin_whpa, axis=0)  # Stack them
        b_low = np.where(big_sum == 0, 1, big_sum)  # Replace 0 values by 1
        b_low = np.flipud(b_low)

        # Save result
        np.save(jp(self.res_dir, 'bin'), b_low)

# %% Hausdorff
    def mhd(self):
        """
        Computes the Modified Hausdorff Distance between the true WHPA that has been recovered from its n first PCA
        components to allow proper comparison.
        """

        # The new idea is to compute MHD with the observed WHPA recovered from it's n first PC.
        n_cut = self.h_pco.n_pc_cut  # Number of components to keep
        # Inverse transform and reshape
        v_h_true_cut = \
            self.h_pco.custom_inverse_transform(self.h_pco.predict_pc, n_cut).reshape((self.shape[1], self.shape[2]))

        # Reminder: these are the focus parameters around the pumping well
        nrow, ncol, x, y = refine_machine(self.x_lim,
                                          self.y_lim,
                                          self.grf)
        # Delineation vertices of the true array
        v_h_true = contours_vertices(x=x,
                                     y=y,
                                     arrays=v_h_true_cut)[0]

        # Compute MHD between the 'true vertices' and the n sampled vertices
        mhds = np.array([modified_hausdorff(v_h_true, vt) for vt in self.vertices])

        # Save mhd
        np.save(jp(self.res_dir, 'haus'), mhds)
