import os
import shutil

import flopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

cols = ['b', 'g', 'r', 'c', 'm']  # Color list


def datread(file=None, header=0):
    """Reads space separated dat file"""
    with open(file, 'r') as fr:
        op = np.array([list(map(float, l.split())) for l in fr.readlines()[header:]])
    return op


def dirmaker(dird):
    """
    Given a folder path, check if it exists, and if not, creates it
    :param dird: path
    :return:
    """
    try:
        if not os.path.exists(dird):
            os.makedirs(dird)
    except:
        pass


def refine_axis(widths, r_pt, ext, cnd, d_dim, a_lim):
    x0 = widths
    x0s = np.cumsum(x0)  # Cumulative sum of the width of the cells
    pt = r_pt
    extx = ext
    cdrx = cnd
    dx = d_dim
    xlim = a_lim

    # X range of the polygon
    xrp = [pt - extx, pt + extx]

    wherex = np.where((xrp[0] < x0s) & (x0s <= xrp[1]))[0]

    # The algorithm must choose a 'flexible parameter', either the cell grid size, the dimensions of the grid or the
    # refined cells themselves
    exn = np.sum(x0[wherex])  # x-extent of the refinement zone
    fx = exn / cdrx  # divides the extent by the new cell spacing
    rx = exn % cdrx  # remainder
    if rx == 0:
        nwxs = np.ones(int(fx)) * cdrx
        x0 = np.delete(x0, wherex)
        x0 = np.insert(x0, wherex[0], nwxs)
    else:  # If the cells can not be exactly subdivided into the new cell dimension
        nwxs = np.ones(int(round(fx))) * cdrx  # Produce a new width vector
        x0 = np.delete(x0, wherex)  # Delete old cells
        x0 = np.insert(x0, wherex[0], nwxs)  # insert new

        cs = np.cumsum(x0)  # Cumulation of width should equal xlim, but it will not be the case, have to adapt width
        difx = xlim - cs[-1]
        where_default = np.where(abs(x0 - dx) <= 5)[0]  # Location of cells whose widths will be adapted
        where_left = where_default[np.where(where_default < wherex[0])]  # Where do we have the default cell size on the
        # left
        where_right = where_default[np.where((where_default >= wherex[0] + len(nwxs)))]  # And on the right
        lwl = len(where_left)
        lwr = len(where_right)

        if lwl > lwr:
            rl = lwl / lwr  # Weights how many cells are on either sides of the refinement zone
            dal = difx / ((lwl + lwr) / lwl)  # Splitting the extra widths on the left and right of the cells
            dal = dal + (difx - dal) / rl
            dar = difx - dal
        elif lwr > lwl:
            rl = lwr / lwl  # Weights how many cells are on either sides of the refinement zone
            dar = difx / ((lwl + lwr) / lwr)  # Splitting the extra widths on the left and right of the cells
            dar = dar + (difx - dar) / rl
            dal = difx - dar
        else:
            dal = difx / ((lwl + lwr) / lwl)  # Splitting the extra widths on the left and right of the cells
            dar = difx - dal

        x0[where_left] = x0[where_left] + dal / lwl
        x0[where_right] = x0[where_right] + dar / lwr

    return x0  # Flip to correspond to flopy expectations


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


def load_flow_model(nam_file, exe_name='', model_ws=''):
    flow_loader = flopy.modflow.mf.Modflow.load
    return flow_loader(f=nam_file, exe_name=exe_name, model_ws=model_ws)


def load_transport_model(nam_file, modflowmodel, exe_name='', model_ws='', ftl_file='mt3d_link.ftl',
                         version='mt3d-usgs'):
    transport_loader = flopy.mt3d.Mt3dms.load
    transport_reloaded = transport_loader(f=nam_file, version=version, modflowmodel=modflowmodel,
                                          exe_name=exe_name, model_ws=model_ws)
    transport_reloaded.ftlfilename = ftl_file
    return transport_reloaded


def evr_plot(pca):
    plt.plot(np.arange(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.show()


def all_curves_plot(curves):
    for j in range(len(curves[0])):
        plt.plot(curves[0][j][:, 0], curves[0][j][:, 1], cols[j], label='c{}'.format(j + 1), alpha=0.5)
    plt.legend()
    for w in range(1, len(curves)):
        for j in range(len(curves[0])):
            plt.plot(curves[w][j][:, 0], curves[w][j][:, 1], cols[j], label='c{}'.format(j + 1), alpha=0.5)
    plt.show()
    plt.close()


def individual_curves_plot(curves):
    for i in range(len(curves[0])):
        for j in range(len(curves)):
            plt.plot(curves[j][i][:, 0], curves[j][i][:, 1], cols[i], label='c{}'.format(i + 1), alpha=0.5)
        plt.legend()
        plt.show()
        plt.close()


def cca_plot(cca_coef, cc, sdc, obs_n, j, x_c_pred, y_c_true):
    plt.plot(cc[:, j], sdc[:, j], 'ro')
    plt.axvline(x_c_pred[obs_n, j])
    plt.plot(x_c_pred[obs_n, j], y_c_true[obs_n, j], 'go')
    plt.xlabel('$d_{}$'.format(j))
    plt.ylabel('$h_{}$'.format(j))
    plt.title('corr = {}'.format(round(cca_coef[j], 3)))


def cca_plots(coeffs, d_cca_t, h_cca_t, d_cca_p, h_cca_p):
    n_comp = len(coeffs)
    fig, axs = plt.subplots(nrows=n_comp % (3 % n_comp) + 1,
                            ncols=3 % n_comp + 1,
                            figsize=(9, 6),
                            subplot_kw={'xticks': [], 'yticks': []})
    for i in range(n_comp):
        axs.flat[i].plot(d_cca_t[i], h_cca_t[i], 'ro')
        axs.flat[i].plot(d_cca_p[i], h_cca_p[i], 'ko')
        axs.flat[i].set_title(str(np.round(coeffs[i], 3)))
    plt.tight_layout()
    plt.show()


class MyWorkshop:

    def __init__(self):
        self.xlim = 1500
        self.ylim = 1000
        self.grf = 1
        self.ntstp = 1000
        self.res_dir = ''

    def load_data(self):
        bkt_files = []
        sd_files = []
        hk_files = []
        # r=root, d=directories, f = files
        roots = []
        for r, d, f in os.walk(self.res_dir, topdown=False):
            if 'bkt.npy' in f and 'sd.npy' in f and 'hk0.npy' in f:
                bkt_files.append(os.path.join(r, 'bkt.npy'))
                sd_files.append(os.path.join(r, 'sd.npy'))
                hk_files.append(os.path.join(r, 'hk0.npy'))
                roots.append(r)
            else:
                try:
                    if r != self.res_dir:
                        shutil.rmtree(r)
                except TypeError:
                    pass

        # Load and filter results
        tpt = list(map(np.load, bkt_files))
        # FIXME: the way transport arrays are saved
        # n_wells = 50
        # tpt = [np.split(t, n_wells) for t in tpt]

        rm = []
        for i in range(len(tpt)):
            for j in range(len(tpt[i])):
                if max(tpt[i][j][:, 1]) > 1:  # Check results files whose max computed head is > 1 and removes them
                    rm.append(i)
                    break
        for index in sorted(rm, reverse=True):
            del bkt_files[index]
            del sd_files[index]
            del hk_files[index]
            del roots[index]

        tpt = list(map(np.load, bkt_files))  # Re-load transport curves
        # tpt = [np.split(t, n_wells) for t in tpt]

        # ny = []  # Smoothed curves
        # tstp = []  # Time steps, which are different for each simulation
        #
        # for c in tpt:  # First, smoothing of the curves with Savgol filter
        #     for ob in c:
        #         tstp.append(c[0][:, 0])
        #         v = ob[:, 1]
        #         ny.append(np.array(savgol_filter(v, 51, 3)))  # smoothed res
        #
        # flat_ny = [item for sublist in ny for item in sublist]
        # max_v = max(flat_ny)  # Max and min value to normalize
        # min_v = min(flat_ny)
        #
        # f1d = []  # List of interpolating functions for each curve
        # for i in range(len(ny)):
        #     o_s = (ny[i] - min_v) / (max_v - min_v)
        #     f1d.append(interp1d(tstp[i], o_s, fill_value='extrapolate'))
        #
        # # ntstp = 1000  # Arbitrary number of time steps to create the final transport array
        # ls = np.linspace(0, 200, num=self.ntstp)  # From 0 to 200 days with 1000 steps
        #
        # tc = []
        # [tc.append(f1d[i](ls)) for i in range(len(f1d))]
        # tc = np.array(tc)
        # tc = tc.reshape((len(tpt), len(tpt[0]), self.ntstp))

        sd = np.array(list(map(np.load, sd_files)))  # Load signed distance
        hk = np.array(list(map(np.load, hk_files)))  # Load hydraulic K

        # return tpt, tc, max_v, min_v, sd, hk
        return tpt, sd, hk

    def protection_zone_plot(self, pz):
        """Displays every protection zones (0 contour line) contained in the pz array"""
        X_pz, Y_pz = np.meshgrid(np.linspace(0, self.xlim, int(self.xlim / self.grf)),
                                 np.linspace(0, self.ylim, int(self.ylim / self.grf)))
        for s in range(len(pz)):
            plt.contour(X_pz, Y_pz, pz[s], [0], colors='blue', alpha=0.5)
        # plt.imshow(Z, origin='lower',
        #            cmap='coolwarm', alpha=0.5)
        # plt.colorbar()
        # plt.xlim(600, 1200)
        # plt.ylim(200, 800)
        # plt.savefig(jp(cwd, 'signed_distance_c.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def hydraulic_conductivity_plot(self, hka):
        for k in range(len(hka)):
            # t = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
            hkp = plt.imshow(hka[k][0], extent=(0, self.xlim, 0, self.ylim), cmap='coolwarm',
                             norm=LogNorm(vmin=hka[k][0].min(), vmax=hka[k][0].max()))
            plt.colorbar()
            # contours = plt.contour(X, Y, sd[i], [0], colors='black', alpha=0.7)
            # plt.savefig(jp(res_dir, 'signed_distance_{}.png'.format(i)), dpi=150, bbox_inches='tight')
            plt.show()
            plt.close()
