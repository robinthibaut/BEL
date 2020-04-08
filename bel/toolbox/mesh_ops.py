import numpy as np


class MeshOps:

    def __init__(self):
        self.xlim = 1500
        self.ylim = 1000
        self.grf = 1  # Cell dimension (1m)
        self.nrow = self.ylim // self.grf
        self.ncol = self.xlim // self.grf

    @staticmethod
    def blocks_from_rc(rows, columns):
        """
        Returns the blocks forming a 2D grid whose rows and columns widths are defined by the two arrays rows, columns
        """

        nrow = len(rows)
        ncol = len(columns)
        delr = rows
        delc = columns
        r_sum = np.cumsum(delr)
        c_sum = np.cumsum(delc)

        blocks = []
        for c in range(nrow):
            for n in range(ncol):
                b = [[c_sum[n] - delc[n], r_sum[c] - delr[c]],
                     [c_sum[n] - delc[n], r_sum[c]],
                     [c_sum[n], r_sum[c]],
                     [c_sum[n], r_sum[c] - delr[c]]]
                blocks.append(b)
        blocks = np.array(blocks)

        return blocks

    @staticmethod
    def blocks_from_rc_3d(rows, columns):
        """
        Returns the blocks forming a 2D grid whose rows and columns widths are defined by the two arrays rows, columns
        """

        nrow = len(rows)
        ncol = len(columns)
        delr = rows
        delc = columns
        r_sum = np.cumsum(delr)
        c_sum = np.cumsum(delc)

        blocks = []
        for c in range(nrow):
            for n in range(ncol):
                b = [[c_sum[n] - delc[n], r_sum[c] - delr[c], 0.],
                     [c_sum[n] - delc[n], r_sum[c], 0.],
                     [c_sum[n], r_sum[c], 0.],
                     [c_sum[n], r_sum[c] - delr[c], 0.]]
                blocks.append(b)
        blocks = np.array(blocks)

        return blocks

    @staticmethod
    def rc_from_blocks(blocks):
        """
        Computes the x and y dimensions of each block
        :param blocks:
        :return:
        """
        dc = np.array([np.diff(b[:, 0]).max() for b in blocks])
        dr = np.array([np.diff(b[:, 1]).max() for b in blocks])

        return dc, dr

    @staticmethod
    def refine_axis(widths, r_pt, ext, cnd, d_dim, a_lim):
        # TODO: write better documentation for this
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

            cs = np.cumsum(
                x0)  # Cumulative width should equal x_lim, but it will not be the case, have to adapt width
            difx = xlim - cs[-1]
            where_default = np.where(abs(x0 - dx) <= 5)[0]  # Location of cells whose widths will be adapted
            where_left = where_default[
                np.where(where_default < wherex[0])]  # Where do we have the default cell size on the
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

    @staticmethod
    def blockshaped(arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array should look like n sub-blocks with
        each sub-block preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        assert h % nrows == 0, "{} rows is not evenly divisible by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisible by {}".format(w, ncols)

        return (arr.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

    def h_sub(self, h, un, uc, sc):
        h_u = np.zeros((h.shape[0], un, uc))
        for i in range(h.shape[0]):
            sim = h[i]
            sub = self.blockshaped(sim, sc, sc)
            h_u[i] = np.array([s.mean() for s in sub]).reshape(un, uc)

        return h_u
