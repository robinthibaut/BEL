#  Copyright (c) 2020. Robin Thibaut, Ghent University

import math
import operator
from functools import reduce
from os.path import join as jp

import flopy
import numpy as np
import vtk
from flopy.export import vtk as vtk_flow

from experiment.base.inventory import MySetup
import experiment.grid.meshio as mops
import experiment.toolbox.filesio as fops


def order_vertices(vertices):
    """
    Paraview expects vertices in a particular order, with the origin at the bottom left corner.
    :param vertices: (x, y) coordinates of the polygon vertices
    :return:
    """
    # Compute center of vertices
    center = \
        tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), vertices), [len(vertices)] * 2))

    # Sort vertices according to angle
    so = sorted(vertices,
                key=lambda coord: (math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))))

    return np.array(so)


class ModelVTK:
    """
    Loads flow/transport models and export the VTK objects.
    """
    def __init__(self, base=None, folder=None):
        self.base = base
        md = self.base.Directories()
        self.rn = folder
        self.bdir = md.main_dir
        self.results_dir = jp(md.hydro_res_dir, self.rn)
        self.vtk_dir = jp(self.results_dir, 'vtk')
        fops.dirmaker(self.vtk_dir)

        # %% Load flow model
        try:
            m_load = jp(self.results_dir, 'whpa.nam')
            self.flow_model = fops.load_flow_model(m_load, model_ws=self.results_dir)
            delr = self.flow_model.modelgrid.delr  # thicknesses along rows
            delc = self.flow_model.modelgrid.delc  # thicknesses along column
            # xyz_vertices = self.flow_model.modelgrid.xyzvertices
            # blocks2d = mops.blocks_from_rc(delc, delr)
            self.blocks = mops.blocks_from_rc_3d(delc, delr)
            # blocks3d = self.blocks.reshape(-1, 3)
        except Exception as e:
            print(e)

        try:
            # Transport model
            mt_load = jp(self.results_dir, 'whpa.mtnam')
            self.transport_model = fops.load_transport_model(mt_load, self.flow_model, model_ws=self.results_dir)
            ucn_files = [jp(self.results_dir, f'MT3D00{i}.UCN') for i in
                         MySetup.Wells.combination]  # Files containing concentration
            ucn_obj = [flopy.utils.UcnFile(uf) for uf in ucn_files]  # Load them
            self.times = [uo.get_times() for uo in ucn_obj]  # Get time steps
            self.concs = np.array([uo.get_alldata() for uo in ucn_obj])  # Get all data
        except Exception as e:
            print(e)

    def flow_vtk(self):
        """
        Export flow model packages and computed heads to vtk format

        """
        dir_fv = jp(self.results_dir, 'vtk', 'flow')
        fops.dirmaker(dir_fv)
        self.flow_model.export(dir_fv, fmt='vtk')
        vtk_flow.export_heads(self.flow_model,
                              jp(self.results_dir, 'whpa.hds'),
                              jp(self.results_dir, 'vtk', 'flow'),
                              binary=True, kstpkper=(0, 0))

    # %% Load transport model

    def transport_vtk(self):
        """
        Export transport package attributes to vtk format

        """
        dir_tv = jp(self.results_dir, 'vtk', 'transport')
        fops.dirmaker(dir_tv)
        self.transport_model.export(dir_tv, fmt='vtk')

    # %% Export UCN to vtk

    def stacked_conc_vtk(self):
        """Stack component concentrations for each time step and save vtk"""
        conc_dir = jp(self.results_dir, 'vtk', 'transport', 'concentration')
        fops.dirmaker(conc_dir)
        # First replace 1e+30 value (inactive cells) by 0.
        conc0 = np.abs(np.where(self.concs == 1e30, 0, self.concs))
        # Cells configuration for 3D blocks
        cells = [("quad", np.array([list(np.arange(i * 4, i * 4 + 4))])) for i in range(len(self.blocks))]
        for j in range(self.concs.shape[1]):
            # Stack the concentration of each component at each time step to visualize them in one plot.
            array = np.zeros(self.blocks.shape[0])
            dic_conc = {}
            for i in range(1, 7):
                # fliplr is necessary as the reshape modifies the array structure
                cflip = np.fliplr(conc0[i - 1, j]).reshape(-1)
                dic_conc['conc_wel_{}'.format(i)] = cflip
                array += cflip  # Stack all components
            dic_conc['stack'] = array

    def conc_vtk(self):
        """Stack component concentrations for each time step and save vtk"""
        # First replace 1e+30 value (inactive cells) by 0.
        conc0 = np.abs(np.where(self.concs == 1e30, 0, self.concs))

        # Initiate points and ugrid
        points = vtk.vtkPoints()
        ugrid = vtk.vtkUnstructuredGrid()

        for e, b in enumerate(self.blocks):
            sb = sorted(b, key=lambda k: [k[1], k[0]])  # Order vertices in vtkPixel convention
            [points.InsertPoint(e * 4 + es, bb) for es, bb in
             enumerate(sb)]  # Insert points by giving first their index e*4+es
            ugrid.InsertNextCell(vtk.VTK_PIXEL, 4, list(range(e * 4, e * 4 + 4)))  # Insert cell in UGrid

        ugrid.SetPoints(points)  # Set points

        for i in range(1, 7):  # For eaxh injecting well
            conc_dir = jp(self.results_dir, 'vtk', 'transport', '{}_UCN'.format(i))
            fops.dirmaker(conc_dir)
            # Initiate array and give it a name
            concArray = vtk.vtkDoubleArray()
            concArray.SetName(f"conc{i}")
            for j in range(self.concs.shape[1]):  # For each time step
                # Set array
                array = np.fliplr(conc0[i - 1, j]).reshape(-1)
                [concArray.InsertNextValue(s) for s in array]
                ugrid.GetCellData().AddArray(concArray)  # Add array to unstructured grid

                # Save grid
                writer = vtk.vtkXMLUnstructuredGridWriter()
                writer.SetInputData(ugrid)
                writer.SetFileName(jp(conc_dir, '{}_conc.vtu'.format(j)))
                writer.Write()

                # Clear storage but keep name
                concArray.Initialize()
                ugrid.GetCellData().Initialize()

    # %% Plot modpath

    def particles_vtk(self, path=1):
        """
        Export travelling particles time series in VTP format
        :param path: Flag to export path's vtk
        :return:
        """
        back_dir = jp(self.results_dir, 'vtk', 'backtrack')
        fops.dirmaker(back_dir)

        # mp_reloaded = backtrack(flowmodel=flow_model, exe_name='', load=True)
        # load the endpoint data
        # endfile = jp(results_dir, 'whpa_mp.mpend')
        # endobj = flopy.utils.EndpointFile(endfile)
        # ept = endobj.get_alldata()

        # # load the pathline data
        # The same information is stored in the time series file
        # pthfile = jp(results_dir, 'whpa_mp.mppth')
        # pthobj = flopy.utils.PathlineFile(pthfile)
        # plines = pthobj.get_alldata()

        # load the time series
        tsfile = jp(self.results_dir, 'whpa_mp.timeseries')
        tso = flopy.utils.modpathfile.TimeseriesFile(tsfile)
        ts = tso.get_alldata()

        n_particles = len(ts)
        n_t_stp = ts[0].shape[0]
        time_steps = ts[0].time

        points_x = np.array([ts[i].x for i in range(len(ts))])
        points_y = np.array([ts[i].y for i in range(len(ts))])
        points_z = np.array([ts[i].z for i in range(len(ts))])

        xs = points_x[:, 0]  # Data at first time step
        ys = points_y[:, 0]
        zs = points_z[:, 0] * 0  # Replace elevation by 0 to project them in the surface
        prev = \
            np.vstack((xs, ys, zs)).T.reshape(-1, 3)

        speed_array = None

        for i in range(n_t_stp):  # For each time step i
            # Get all particles positions
            xs = points_x[:, i]
            ys = points_y[:, i]
            zs = points_z[:, i] * 0  # Replace elevation by 0 to project them in the surface
            xyz_particles_t_i = \
                np.vstack((xs, ys, zs)).T.reshape(-1, 3)

            # Compute instant speed ds/dt
            speed = np.abs(operator.truediv(tuple(map(np.linalg.norm, xyz_particles_t_i - prev)),
                                            time_steps[i] - time_steps[i - 1]))
            prev = xyz_particles_t_i

            if speed_array is None:
                speed_array = np.array([speed])
            else:
                speed_array = np.append(speed_array, np.array([speed]), 0)

            # Initiate points object
            points = vtk.vtkPoints()
            ids = [points.InsertNextPoint(c) for c in xyz_particles_t_i]

            # Create a cell array to store the points
            vertices = vtk.vtkCellArray()
            vertices.InsertNextCell(n_particles)
            [vertices.InsertCellPoint(ix) for ix in ids]

            # Create a polydata to store everything in
            polyData = vtk.vtkPolyData()
            # Add the points to the dataset
            polyData.SetPoints(points)
            polyData.SetVerts(vertices)

            # Assign value array
            speedArray = vtk.vtkDoubleArray()
            speedArray.SetName("Speed")
            [speedArray.InsertNextValue(s) for s in speed]
            polyData.GetPointData().AddArray(speedArray)

            # Write data
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetInputData(polyData)
            writer.SetFileName(jp(back_dir, 'particles_t{}.vtp'.format(i)))
            writer.Write()

            # Write path lines
            if i and path:
                for p in range(n_particles):
                    short = np.vstack((points_x[p, :i + 1], points_y[p, :i + 1], np.abs(points_z[p, :i + 1] * 0))).T
                    points = vtk.vtkPoints()
                    [points.InsertNextPoint(c) for c in short]

                    # Create a polydata to store everything in
                    polyData = vtk.vtkPolyData()
                    # Add the points to the dataset
                    polyData.SetPoints(points)

                    # Create a cell array to store the lines in and add the lines to it
                    cells = vtk.vtkCellArray()
                    cells.InsertNextCell(i + 1)
                    [cells.InsertCellPoint(k) for k in range(i + 1)]

                    # Create value array and assign it to the polydata
                    speed = vtk.vtkDoubleArray()
                    speed.SetName("speed")
                    [speed.InsertNextValue(speed_array[k][p]) for k in range(i + 1)]
                    polyData.GetPointData().AddArray(speed)

                    # Add the lines to the dataset
                    polyData.SetLines(cells)

                    # Export
                    writer = vtk.vtkXMLPolyDataWriter()
                    writer.SetInputData(polyData)
                    writer.SetFileName(jp(back_dir, 'path{}_t{}.vtp'.format(p, i)))
                    writer.Write()
            else:
                for p in range(n_particles):
                    xs = points_x[p, i]
                    ys = points_y[p, i]
                    zs = points_z[p, i] * 0  # Replace elevation by 0 to project them in the surface
                    xyz_particles_t_i = \
                        np.vstack((xs, ys, zs)).T.reshape(-1, 3)

                    # Create points
                    points = vtk.vtkPoints()
                    ids = [points.InsertNextPoint(c) for c in xyz_particles_t_i]

                    # Create a cell array to store the points
                    vertices = vtk.vtkCellArray()
                    vertices.InsertNextCell(len(xyz_particles_t_i))  # Why is this line necessary ?
                    [vertices.InsertCellPoint(ix) for ix in ids]

                    # Create a polydata to store everything in
                    polyData = vtk.vtkPolyData()
                    # Add the points to the dataset
                    polyData.SetPoints(points)
                    polyData.SetVerts(vertices)

                    # Export
                    writer = vtk.vtkXMLPolyDataWriter()
                    writer.SetInputData(polyData)
                    writer.SetFileName(jp(back_dir, 'path{}_t{}.vtp'.format(p, i)))
                    writer.Write()

    # %% Export wells objects as vtk

    def wells_vtk(self):
        """Exports wells coordinates to VTK"""

        wbd = self.base.Wells().wells_data

        wels = np.array([wbd[o]['coordinates'] for o in wbd])
        wels = np.insert(wels, 2, np.zeros(len(wels)), axis=1)  # Insert zero array for Z

        # Export wells as VTK points
        points = vtk.vtkPoints()  # Points
        ids = [points.InsertNextPoint(w) for w in wels]  # Points IDS
        welArray = vtk.vtkCellArray()  # Vertices
        welArray.InsertNextCell(len(wels))
        [welArray.InsertCellPoint(ix) for ix in ids]
        welPolydata = vtk.vtkPolyData()  # PolyData to store everything
        welPolydata.SetPoints(points)
        welPolydata.SetVerts(welArray)
        # Save objects
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(welPolydata)
        writer.SetFileName(jp(self.vtk_dir, 'wells.vtp'))
        writer.Write()
