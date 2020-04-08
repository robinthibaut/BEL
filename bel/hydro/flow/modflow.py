import os
from datetime import date
from os.path import join as jp

import flopy
import flopy.utils.binaryfile as bf
import numpy as np
from scipy.spatial import distance_matrix

from bel.toolbox.sgems.sgems import SGEMS
from bel.toolbox.mesh_ops import MeshOps


def flow(exe_name, model_ws, wells):
    # Model name
    model_name = 'whpa'
    grid_dir = jp(os.getcwd(), 'grid')  # Directory containing grid information
    # %% Modflow
    model = flopy.modflow.Modflow(modelname=model_name,
                                  namefile_ext='nam',
                                  version='mf2005',
                                  exe_name=exe_name,
                                  structured=True,
                                  listunit=2,
                                  model_ws=model_ws,
                                  verbose=False)

    # %% Model time discretization

    itmuni = 4  # Time units = days
    nper = 3  # Number of stress periods
    perlen = [1, 1 / 12, 100]  # Period lengths in days
    nstp = [1, 300, 100]  # Number of time steps per time period
    tsmult = [1, 1, 1.1]  # Time multiplier for each period
    steady = [True, False, False]  # State flags for each period
    td = date.today()  # Start date of the simulation
    start_datetime = td.strftime('%Y-%m-%d')  # Start of simulation = today

    # %% Model dimensions

    lenuni = 2  # Distance units = meters

    x_lim = 1500
    y_lim = 1000

    dx = 10  # Block x-dimension
    dy = 10  # Block y-dimension
    dz = 10  # Block z-dimension

    nrow = int(y_lim / dy)  # Number of rows
    ncol = int(x_lim / dx)  # Number of columns
    nlay = 1  # Number of layers

    xo = 0  # Grid x origin
    yo = 0  # Grid y origin
    zo = 0  # Grid z origin

    # Refinement
    # pt is the first instance of the pumping well location.
    pw_d = np.load(jp(model_ws, 'pw.npy'), allow_pickle=True)  # Pumping well location
    # Point around which refinement will occur
    pt = pw_d[0][:2]
    # [cell size, extent around pt in m]
    r_params = np.array([[9, 150],
                         [8, 100],
                         [7, 90],
                         [6, 80],
                         [5, 70],
                         [4, 60],
                         [3, 50],
                         [2.5, 40],
                         [2, 30],
                         [1.5, 20],
                         [1, 10]])

    # Saving r_params to avoid computing distance matrix each time
    flag_dis = 0  # If discretization is different
    disf = jp(grid_dir, 'dis.txt')  # discretization txt file
    if os.path.exists(disf):
        r_params_loaded = np.loadtxt(disf)  # loads dis info
        if not np.array_equal(r_params, r_params_loaded):  # if new refinement parameters differ from the previous one
            np.savetxt(disf, r_params)  # update file
        else:
            flag_dis = 1  # If discretization is the same, use old distance matrix
    else:
        np.savetxt(disf, r_params)

    delc = np.ones(ncol) * dx  # Size of each cell in x-dimension - columns
    delr = np.ones(nrow) * dy  # Size of each cell in y-dimension - rows

    # Refinement of the mesh
    r_a = MeshOps.refine_axis
    for p in r_params:
        delc = r_a(delc, pt[0], p[1], p[0], dx, x_lim)
        delr = r_a(delr, pt[1], p[1], p[0], dy, y_lim)

    ncol = len(delc)  # new number of columns
    nrow = len(delr)

    top = 0  # Model top (m)
    botm = [-dz]  # List containing bottom location (m) of each layer

    # %% Model initial conditions

    ibound = np.ones((nlay, nrow, ncol), dtype=np.int)  # Active cells
    ibound[0, :, 0] = -1  # Fixed left side - set value to -1
    ibound[0, :, -1] = -1  # Fixed right side

    strt = np.zeros((nlay, nrow, ncol), dtype=np.float)  # Starting heads at 0
    h1 = -3  # Water table level at the right side
    strt[0, :, -1] = h1  # Fixing the value, that will remain constant during the stress period

    # %% ModflowDis

    min_cell_dim = min(dx, dy)
    nrow_d = int(y_lim / min_cell_dim)
    ncol_d = int(x_lim / min_cell_dim)

    # Creation of a dummy modflow grid to work with sgems, which doesn't have the same cell structure.
    model_dummy = flopy.modflow.Modflow(modelname='dummy')

    dis_sgems = flopy.modflow.ModflowDis(model=model_dummy,
                                         nlay=nlay,
                                         nrow=nrow_d,
                                         ncol=ncol_d,
                                         delr=min_cell_dim,
                                         delc=min_cell_dim,
                                         top=top,
                                         botm=botm)
    # Nodes coordinates of dummy model
    ncd0 = dis_sgems.get_node_coordinates()

    xyz_dummy = []
    # Fill array with x,y coordinates of cell centers
    for yc in ncd0[0]:
        for xc in ncd0[1]:
            xyz_dummy.append([xc, yc])
    # TODO: delr and delc seem to be inverted - write an issue
    dis5 = flopy.modflow.ModflowDis(model=model,
                                    nlay=nlay,
                                    nrow=nrow,
                                    ncol=ncol,
                                    nper=nper,
                                    delr=delc,
                                    delc=delr,
                                    laycbd=0,
                                    top=top,
                                    botm=botm,
                                    perlen=perlen,
                                    nstp=nstp,
                                    tsmult=tsmult,
                                    steady=steady,
                                    itmuni=itmuni,
                                    lenuni=lenuni,
                                    extension='dis',
                                    rotation=0.0,
                                    start_datetime=start_datetime)

    ncd1 = dis5.get_node_coordinates()  # Get y, x, and z cell centroids of dummy model
    xyz_true = []
    for yc in ncd1[0]:
        for xc in ncd1[1]:
            xyz_true.append([xc, yc])

    def get_node_id(dis, x, y):
        # First get rc
        node_rc = dis.get_rc_from_node_coordinates(x, y)  # Get node number of the wel location
        # Then extract node
        node = dis.get_node((0,) + node_rc)[0]
        return node

    def make_well(wel_info):
        """
        Produce well stress period data readable by modflow
        :param wel_info: [ r, c, [rate sp #0, ..., rate sp# n] ]
        :return:
        """
        iw = [0, wel_info[0], wel_info[1]]  # Injection wel location - layer row column
        iwr = wel_info[2]  # Well rate for the defined time periods
        iw_lrc = [0] + list(dis5.get_rc_from_node_coordinates(iw[1], iw[2]))
        spiw = [iw_lrc + [r] for r in iwr]  # Defining list containing stress period data under correct format
        return [iw, iwr, iw_lrc, spiw]

    pwa = pw_d  # Pumping well location
    iwa = np.array(wells)  # wells is given as function argument

    wells_data = np.concatenate((pwa, iwa), axis=0)  # Concatenates well data (pumping, injection)

    my_wells = [make_well(o) for o in wells_data]  # Produce well stress period data readable by modflow

    spd = [mw[-1] for mw in my_wells]  # Collecting SPD for each well
    spd = np.array(spd)

    np.save(jp(model_ws, 'spd'), spd)  # Saves the SPD

    # %% ModflowWel

    wel_stress_period_data = {}

    for sp in range(nper):
        wel_stress_period_data[sp] = np.array(spd)[:, sp]

    # stress_period_data =
    # {
    # 0: [[    0 ,    50 ,    80 , -2400. ]],
    # 1: [[    0 ,    50 ,    80 , -2400. ]],
    # 2: [[    0 ,    50 ,    80 , -2400. ]]
    # }

    wel = flopy.modflow.ModflowWel(model=model,
                                   stress_period_data=wel_stress_period_data)

    # wel.check(level=0)

    # %% ModflowBas

    bas = flopy.modflow.ModflowBas(model=model,
                                   ibound=ibound,
                                   strt=strt,
                                   ifrefm=True,
                                   ixsec=False,
                                   ichflg=False,
                                   stoper=None,
                                   hnoflo=-999.99,
                                   extension='bas',
                                   unitnumber=None,
                                   filenames=None)

    # bas.check(level=0)

    # %% ModflowPcg

    pcg = flopy.modflow.ModflowPcg(model=model,
                                   mxiter=70,
                                   iter1=70,
                                   npcond=1,
                                   hclose=1e-3,
                                   rclose=1e-3,
                                   relax=0.99,
                                   nbpol=0,
                                   iprpcg=0,
                                   mutpcg=3,
                                   damp=1.0,
                                   dampt=1.0,
                                   ihcofadd=0,
                                   extension='pcg',
                                   unitnumber=None,
                                   filenames=None)

    # %% ModflowOc

    stress_period_data_oc = {}

    for kper in range(nper):  # For each stress period
        for kstp in range(nstp[kper]):  # For each time step per time period
            # MT3DMS needs HEAD
            # MODPATH needs BUDGET
            stress_period_data_oc[(kper, kstp)] = ['save head',
                                                   'print head',
                                                   'save budget',
                                                   'print budget']

    oc = flopy.modflow.ModflowOc(model=model,
                                 stress_period_data=stress_period_data_oc,
                                 compact=True)

    # %% ModflowLmt
    # Necessary to create a xxx.ftl file to link modflow simulation to mt3dms transport simulation.
    output_file_name = 'mt3d_link.ftl'

    lmt = flopy.modflow.ModflowLmt(model=model,
                                   output_file_name=output_file_name)

    # %% SGEMS

    op = 'hk'  # Simulations output name

    if os.path.exists(jp(model_ws, '{}.npy').format(op)):  # If re-using an older model
        valkr = np.load(jp(model_ws, '{}.npy').format(op))
    else:
        sgems = SGEMS()
        nr = 1  # Number of realizations.
        # Extracts wells nodes number in sgems grid, to fix their simulated value.
        wells_nodes_sgems = [get_node_id(dis_sgems, w[0], w[1]) for w in wells_data]
        # Hard data node
        # fixed_nodes = [[pwnode_sg, 2], [iw1node_sg, 1], [iw2node_sg, 1.5], [iw3node_sg, 0.7], [iw4node_sg, 1.2]]
        # I now arbitrarily attribute a random value between 1 and 2 to the well nodes
        fixed_nodes = [[w, 1 + np.random.rand()] for w in wells_nodes_sgems]

        sgrid = [dis_sgems.ncol,
                 dis_sgems.nrow,
                 dis_sgems.nlay,
                 dis_sgems.delc.array[0],
                 dis_sgems.delr.array[0],
                 dis_sgems.delr.array[0],
                 xo, yo, zo]  # Grid information

        seg = [50, 50, 50, 0, 0, 0]  # Search ellipsoid geometry

        sgems.gaussian_simulation(op_folder=model_ws.replace('\\', '//'),
                                  simulation_name='hk',
                                  output=op,
                                  grid=sgrid,
                                  fixed_nodes=fixed_nodes,
                                  algo='sgsim',
                                  number_realizations=nr,
                                  seed=np.random.randint(1000000),
                                  kriging_type='Simple Kriging (SK)',
                                  trend=[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  local_mean=0,
                                  hard_data_grid='hd',
                                  hard_data_property='hard',
                                  assign_hard_data=1,
                                  max_conditioning_data=15,
                                  search_ellipsoid_geometry=seg,
                                  target_histogram_flag=0,
                                  target_histogram=[0, 0, 0, 0],
                                  variogram_nugget=0,
                                  variogram_number_stuctures=1,
                                  variogram_structures_contribution=[1],
                                  variogram_type=['Spherical'],
                                  range_max=[100],
                                  range_med=[50],
                                  range_min=[25],
                                  angle_x=[0],
                                  angle_y=[0],
                                  angle_z=[0])

        opl = jp(model_ws, '{}.grid'.format(op))  # Output file location.

        hk = SGEMS.so(opl)  # Grid information directly derived from the output file.

        k_mean = np.random.uniform(1.4, 2)  # Hydraulic conductivity mean between x and y in m/d.

        k_std = 0.4

        hkp = np.copy(hk)

        hk_array = [SGEMS.transform(h, k_mean, k_std) for h in hkp]

        # Setting the hydraulic conductivity matrix.
        hk_array = [np.reshape(h, (nlay, dis_sgems.nrow, dis_sgems.ncol)) for h in hk_array]

        # Flip to correspond to sgems grid ! hk_array is now a list of the
        # arrays of conductivities for each realization.
        hk_array = [np.fliplr(hk_array[h]) for h in range(0, nr, 1)]

        np.save(jp(model_ws, op + '0'), hk_array[0])  # Save the un-discretized hk grid

        # Flattening hk_array to plot it
        fl = [item for sublist in hk_array[0] for item in sublist]
        fl2 = [item for sublist in fl for item in sublist]
        val = []
        for n in range(nlay):  # Adding 'nlay' times so all layers get the same conductivity.
            val.append(fl2)
        val = [item for sublist in val for item in sublist]  # Flattening

        # If the sgems grid is different from the modflow grid, which might be the case since we would like to refine
        # in some ways the flow mesh, the piece of code below assigns to the flow grid the values of the hk simulations
        # based on the closest distance between cells.
        inds_file = jp(grid_dir, 'inds.npy')  # Index file location - relates the position of closest cells
        # between differently discretized meshes.
        if nrow_d != nrow or ncol_d != ncol:  # if mismatch between nrow and ncol, that is to say, we must copy/paste
            # the new hk array on a new grid.
            if flag_dis == 0:
                dm = distance_matrix(xyz_true, xyz_dummy)  # Compute distance matrix between refined and dummy grid.
                inds = [np.unravel_index(np.argmin(dm[i], axis=None), dm[i].shape)[0] for i in range(dm.shape[0])]
                np.save(inds_file, inds)  # Save index file to avoid re-computing
            else:  # If the inds file exists.
                inds = np.load(inds_file)
            valk = [val[k] for k in inds]  # Contains k values for refined grid
            valkr = np.reshape(valk, (nlay, nrow, ncol))  # Reshape in n layers x n cells in refined grid.
        else:
            valkr = hk_array[0]

        np.save(jp(model_ws, op), valkr)

    # %% Layer 1 properties

    laytyp = 1
    layavg = 0
    chani = 0
    layvka = 0
    laywet = 0
    ipakcb = 53
    hdry = -1E+30
    iwdflg = 0
    wetfct = 0.1
    iwetit = 1
    ihdwet = 0

    hk = valkr  # Horizontal hydraulic conductivity

    hani = 1.0
    vka = hk / 10  # Vertical hydraulic conductivity
    ss = 10e-4  # Specific storage
    sy = 0.25  # Specific yield
    vkcb = 0.0
    wetdry = -0.01
    storagecoefficient = False
    constantcv = False
    thickstrt = False
    nocvcorrection = False
    novfc = False

    # %% ModflowLpf

    lpf = flopy.modflow.ModflowLpf(model=model,
                                   laytyp=laytyp,
                                   layavg=layavg,
                                   chani=chani,
                                   layvka=layvka,
                                   laywet=laywet,
                                   ipakcb=ipakcb,
                                   hdry=hdry,
                                   iwdflg=iwdflg,
                                   wetfct=wetfct,
                                   iwetit=iwetit,
                                   ihdwet=ihdwet,
                                   hk=hk,
                                   hani=hani,
                                   vka=vka,
                                   ss=ss,
                                   sy=sy,
                                   vkcb=vkcb,
                                   wetdry=wetdry,
                                   storagecoefficient=storagecoefficient,
                                   constantcv=constantcv,
                                   thickstrt=thickstrt,
                                   nocvcorrection=nocvcorrection,
                                   novfc=novfc,
                                   extension='lpf')

    # lpf.check(level=0)

    # %% Modflow Run

    model.write_input()

    # Run the model
    model.run_model(silent=True,
                    pause=False,
                    report=True)

    # model.check(level=0)

    # %% Checking flow results

    headobj = bf.HeadFile(jp(model_ws, '{}.hds'.format(model_name)))  # Create the headfile and budget file objects
    times = headobj.get_times()

    head = headobj.get_data(totim=times[-1])  # Get last data
    headobj.close()

    if head.max() > top + 1:  # Quick check - if the maximum computed head is higher than the layer top, it means
        # that an error occurred, and we shouldn't waste time computing the transport on a false solution.
        # TODO: optimize this
        model = None

    return model
