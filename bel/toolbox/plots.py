from os.path import join as jp
import matplotlib.pyplot as plt
import numpy as np


class Plot:

    def __init__(self, xlim, ylim, grf):

        self.xlim = 1500
        self.ylim = 1000
        self.grf = 5
        self.nrow = self.ylim // self.grf
        self.ncol = self.xlim // self.grf
        self.x, self.y = np.meshgrid(
            np.linspace(0, self.xlim, int(self.xlim / self.grf)), np.linspace(0, self.ylim, int(self.ylim / self.grf)))
        self.wdir = jp('..', 'hydro', 'grid')
        self.cols = ['w', 'g', 'r', 'c', 'm', 'y']
        np.random.shuffle(self.cols)

    def curves(self, tc, n_wel, sdir=None, show=False):
        """
        Shows every breakthrough curve stacked on a plot.
        @param tc: Curves with shape (n_sim, n_wells, n_time_steps)
        @param n_wel: Number of observation points
        @param sdir: Directory in which to save figure
        @param show: Whether to show or not
        """
        for i in range(len(tc)):
            for t in range(n_wel):
                plt.plot(tc[i][t], color=self.cols[t], linewidth=.2, alpha=0.5)
        plt.grid(linewidth=.3, alpha=.4)
        plt.tick_params(labelsize=5)
        if sdir:
            plt.savefig(jp(sdir, 'curves.png'), dpi=300)
            plt.close()
        if show:
            plt.show()
            plt.close()

    def curves_i(self, tc, n_wel, sdir=None, show=False):
        """
        Shows every breakthrough individually for each observation point.
        Will produce n_well figures of n_sim curves each.
        @param tc: Curves with shape (n_sim, n_wells, n_time_steps)
        @param n_wel: Number of observation points
        @param sdir: Directory in which to save figure
        @param show: Whether to show or not
        """
        for t in range(n_wel):
            for i in range(len(tc)):
                plt.plot(tc[i][t], color=self.cols[t], linewidth=.2, alpha=0.5)
            plt.grid(linewidth=.3, alpha=.4)
            plt.tick_params(labelsize=5)
            if sdir:
                plt.savefig(jp(sdir, 'curves_{}.png'.format(t)), dpi=300)
                plt.close()
            if show:
                plt.show()
                plt.close()

    def whp(self,
            h,
            alpha=0.4,
            lw=.5,
            bkg_field_array=None,
            vmin=None,
            vmax=None,
            cmap='coolwarm',
            colors='white',
            fig_file=None,
            show=False):
        """
        Produces the WHPA plot, that is the zero-contour of the signed distance array.
        It assumes that well information can be loaded from pw.npy and iw.npy.
        I should change this.
        @param cmap: colormap for the background array
        @param vmax: max value to plot for the background array
        @param vmin: max value to plot for the background array
        @param bkg_field_array: 2D array whose values will be plotted on the grid
        @param h: Array containing grids of values whose 0 contour will be computed and plotted
        @param alpha: opacity of the 0 contour lines
        @param lw: Line width
        @param colors: Line color
        @param fig_file:
        @param show:
        @return:
        """
        # TODO: Add more options to customize the plot.
        # Plot background
        if bkg_field_array is not None:
            plt.imshow(bkg_field_array,
                       extent=(0, self.xlim, 0, self.ylim),
                       vmin=vmin,
                       vmax=vmax,
                       cmap=cmap)
            plt.colorbar()
        # Plot results
        for z in h:  # h is the n square WHPA matrix
            plt.contour(self.x, self.y, z, [0], colors=colors, linewidths=lw, alpha=alpha)
        plt.grid(color='c', linestyle='-', linewidth=.5, alpha=.2)
        # Plot wells
        pwl = np.load((jp(self.wdir, 'pw.npy')), allow_pickle=True)[:, :2]
        plt.plot(pwl[0][0], pwl[0][1], 'wo', label='pw')
        iwl = np.load((jp(self.wdir, 'iw.npy')), allow_pickle=True)[:, :2]
        for i in range(len(iwl)):
            plt.plot(iwl[i][0], iwl[i][1], 'o', markersize=4, markeredgecolor='k', markeredgewidth=.5,
                     label='iw{}'.format(i))
        plt.legend(fontsize=8)
        plt.xlim(800, 1150)
        plt.ylim(300, 700)
        plt.tick_params(labelsize=5)
        if fig_file:
            plt.savefig(fig_file, bbox_inches='tight', dpi=300)
            plt.close()
        if show:
            plt.show()
            plt.close()

    def whp_prediction(self, forecasts, h_true, h_pred, fig_file=None, show=False):
        self.whp(h=forecasts)
        # Plot true h
        plt.contour(self.x, self.y, h_true, [0], colors='red', linewidths=1, alpha=.9)
        # Plot true h predicted
        plt.contour(self.x, self.y, h_pred, [0], colors='cyan', linewidths=1, alpha=.9)
        if fig_file:
            plt.savefig(fig_file, bbox_inches='tight', dpi=300)
            plt.close()
        if show:
            plt.show()
            plt.close()

    def h_pca_inverse_plot(self, v, e, pca_o, vn):
        """
        Plot used to compare the reproduction of the original physical space after PCA transformation
        @param v: Original, untransformed signed distance array
        @param e: Sample number on which the test is performed
        @param pca_o: signed distance PCA operator
        @param vn: Number of components to inverse-transform.
        @return:
        """
        v_pc = pca_o.transform(v)
        v_pred = (np.dot(v_pc[e, :vn], pca_o.components_[:vn, :]) + pca_o.mean_)
        self.whp(v_pred.reshape(1, self.nrow, self.ncol), colors='cyan', alpha=.8, lw=1, show=False)
        self.whp(v[e].reshape(1, self.nrow, self.ncol), colors='red', alpha=1, lw=1, show=True)

    @staticmethod
    def d_pca_inverse_plot(v, e, pca_o, vn):
        """
        Plot used to compare the reproduction of the original physical space after PCA transformation
        @param v: Original, untransformed data array
        @param e: Sample number on which the test is performed
        @param pca_o: data PCA operator
        @param vn: Number of components to inverse-transform the data
        @return:
        """
        v_pc = pca_o.transform(v)
        v_pred = np.dot(v_pc[e, :vn], pca_o.components_[:vn, :]) + pca_o.mean_
        plt.plot(v[e], 'r', alpha=.8)
        plt.plot(v_pred, 'c', alpha=.8)
        plt.show()

    @staticmethod
    def explained_variance(pca, n_comp=0, xfs=2, fig_file=None, show=False):
        """
        PCA explained variance plot
        @param pca: PCA operator
        @param n_comp: Number of components to display
        @param xfs: X-axis fontsize
        @param fig_file:
        @param show:
        @return:
        """
        plt.grid(alpha=0.2)
        if not n_comp:
            n_comp = pca.n_components_
        plt.xticks(np.arange(n_comp), fontsize=xfs)
        plt.plot(np.arange(n_comp), np.cumsum(pca.explained_variance_ratio_[:n_comp]),
                 '-o', linewidth=.5, markersize=1.5, alpha=.8)
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        if fig_file:
            plt.savefig(fig_file, dpi=300)
            plt.close()
        if show:
            plt.show()
            plt.close()

    @staticmethod
    def pca_scores(training, prediction, n_comp, fig_file=None, show=False):
        """
        PCA scores plot, displays scores of observations above those of training.
        @param training: Training scores
        @param prediction: Test scores
        @param n_comp: How many componnents to show
        @param fig_file:
        @param show:
        @return:
        """
        # Scores plot
        plt.grid(alpha=0.2)
        ut = n_comp
        plt.xticks(np.arange(ut), fontsize=8)
        plt.plot(training.T[:ut], 'wo', markersize=1, alpha=0.2)  # Plot all training scores
        for sample_n in range(len(prediction)):
            pc_obs = prediction[sample_n]
            plt.plot(pc_obs.T[:ut],  # Plot observations scores
                     'o', markersize=2.5, markeredgecolor='k', markeredgewidth=.4, alpha=.8,
                     label=str(sample_n))
        plt.tick_params(labelsize=6)
        plt.legend(fontsize=3)

        if fig_file:
            plt.savefig(fig_file, dpi=300)
            plt.close()
        if show:
            plt.show()
            plt.close()

    @staticmethod
    def cca(cca, d, h, d_pc_prediction, h_pc_prediction, sdir=None, show=False):
        """
        CCA plots.
        Receives d, h PC components to be predicted, transforms them in CCA space and adds it to the plots.
        @param cca: CCA operator
        @param d: d CCA scores
        @param h: h CCA scores
        @param d_pc_prediction: d test PC scores
        @param h_pc_prediction: h test PC scores
        @param sdir:
        @param show:
        @return:
        """

        cca_coefficient = np.corrcoef(d, h).diagonal(offset=cca.n_components)  # Gets correlation coefficient

        # CCA plots for each observation:
        for i in range(cca.n_components):
            comp_n = i
            plt.plot(d[comp_n], h[comp_n], 'ro', markersize=3, markerfacecolor='r', alpha=.25)
            for sample_n in range(len(d_pc_prediction)):  # For each 'observation'
                d_obs = d_pc_prediction[sample_n]
                h_obs = h_pc_prediction[sample_n]
                d_cca_prediction, h_cca_prediction = cca.transform(d_obs.reshape(1, -1),
                                                                   h_obs.reshape(1, -1))
                d_cca_prediction, h_cca_prediction = d_cca_prediction.T, h_cca_prediction.T

                plt.plot(d_cca_prediction[comp_n], h_cca_prediction[comp_n],
                         'o', markersize=4.5, alpha=.7,
                         label='{}'.format(sample_n))

            plt.grid('w', linewidth=.3, alpha=.4)
            plt.tick_params(labelsize=8)
            plt.title(round(cca_coefficient[i], 4))
            plt.legend(fontsize=5)
            if sdir:
                plt.savefig(jp(sdir, 'cca{}.png'.format(i)), bbox_inches='tight', dpi=300)
                plt.close()
            if show:
                plt.show()
                plt.close()
