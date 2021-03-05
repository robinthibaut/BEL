#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
from datetime import date
from os.path import join as jp

import flopy
import flopy.utils.binaryfile as bf
import numpy as np
from scipy.spatial import distance_matrix

from ...spatial import refine_axis
from ...config import Setup


def flow(exe_name: str, model_ws: str, grid_dir: str, hk_array: np.array,
         xy_dummy: np.array):
    """
    Builds and run a customized MODFLOW simulation.
    :param exe_name: Path to the executable file.
    :param model_ws: Path to the working directory.
    :param grid_dir: Path to wells data directory.
    :param hk_array: Hydraulic conductivity array.
    :param xy_dummy: [x, y] coordinates of the centers of the cell of the geostatistical simulation grid.

    :return:
    """
    # Model name
    model_name = Setup.Files.project_name
    # %% Modflow
    model = flopy.modflow.Modflow(
        modelname=model_name,
        namefile_ext="nam",
        version="mf2005",
        exe_name=exe_name,
        structured=True,
        listunit=2,
        model_ws=model_ws,
        verbose=False,
    )

    # %% Model time discretization

    itmuni = 4  # Time units = days
    nper = 3  # Number of stress periods
    perlen = [1, 1 / 12, 100]  # Period lengths in days
    nstp = [1, 300, 100]  # Number of time steps per time period
    tsmult = [1, 1, 1.1]  # Time multiplier for each period
    steady = [True, False, False]  # State flags for each period
    td = date.today()  # Start date of the simulation
    start_datetime = td.strftime("%Y-%m-%d")  # Start of simulation = today

    # %% Model dimensions

    lenuni = 2  # Distance units = meters

    gd = Setup.GridDimensions()

    x_lim = gd.x_lim
    y_lim = gd.y_lim

    dx = gd.dx  # Block x-dimension
    dy = gd.dy  # Block y-dimension
    dz = gd.dz  # Block z-dimension

    nrow = gd.nrow  # Number of rows
    ncol = gd.ncol  # Number of columns
    nlay = gd.nlay  # Number of layers

    # Refinement
    wcd = Setup.Wells()
    pw_d = wcd.wells_data["pumping0"]
    # Point around which refinement will occur
    pt = pw_d["coordinates"]
    # [cell size, extent around pt in m]
    r_params = gd.r_params

    def refine_():
        """
        Refine X-Y axes.
        """
        along_c = np.ones(
            ncol) * dx  # Size of each cell in x-dimension - columns
        along_r = np.ones(nrow) * dy  # Size of each cell in y-dimension - rows
        r_a = refine_axis
        for p in r_params:
            along_c = r_a(along_c, pt[0], p[1], p[0], dx, x_lim)
            along_r = r_a(along_r, pt[1], p[1], p[0], dy, y_lim)
        np.save(jp(grid_dir, "delc"), along_c)
        np.save(jp(grid_dir, "delr"), along_r)

        return along_c, along_r

    # Saving r_params to avoid computing distance matrix each time
    flag_dis = 0  # If discretization is different
    disf = jp(grid_dir, "dis.txt")  # discretization txt file
    if os.path.exists(disf):
        r_params_loaded = np.loadtxt(disf)  # loads dis info
        # if new refinement parameters differ from the previous one
        if not np.array_equal(r_params, r_params_loaded):
            np.savetxt(disf, r_params)  # update file
            delc, delr = refine_()
        else:
            flag_dis = 1  # If discretization is the same, use old distance matrix
            try:
                delc = np.load(jp(grid_dir, "delc.npy"))
                delr = np.load(jp(grid_dir, "delr.npy"))
            except FileNotFoundError:
                delc, delr = refine_()
    else:
        np.savetxt(disf, r_params)
        delc, delr = refine_()

    ncol = len(delc)  # new number of columns
    nrow = len(delr)  # new number of rows

    top = np.zeros((nrow, ncol), dtype=float)  # Model top (m)
    botm = [-dz]  # List containing bottom location (m) of each layer

    # %% Model initial conditions

    ibound = np.ones((nlay, nrow, ncol), dtype=int)  # Active cells
    ibound[0, :, 0] = -1  # Fixed left side - set value to -1
    ibound[0, :, -1] = -1  # Fixed right side

    strt = np.zeros((nlay, nrow, ncol), dtype=float)  # Starting heads at 0
    h1 = -3  # Water table level at the right side
    # Fixing the value, that will remain constant during the stress period
    strt[0, :, -1] = h1

    # %% ModflowDis

    min_cell_dim = min(dx, dy)
    nrow_d = y_lim // min_cell_dim
    ncol_d = x_lim // min_cell_dim

    dis5 = flopy.modflow.ModflowDis(
        model=model,
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
        extension="dis",
        rotation=0.0,
        start_datetime=start_datetime,
    )

    # Get y, x, and z cell centroids of true model grid
    ncd1 = dis5.get_node_coordinates()

    xy_true = []
    for yc in ncd1[0]:
        for xc in ncd1[1]:
            xy_true.append([xc, yc])

    def make_well(well_name: str):
        """
        Produces well stress period data readable by Modflow.
        :param well_name: [ r, c, [rate sp #0, ..., rate sp# n] ]
        """
        iw = [
            0,
            wcd.wells_data[well_name]["coordinates"][0],
            wcd.wells_data[well_name]["coordinates"][1],
        ]
        # Well rate for the defined time periods
        iwr = wcd.wells_data[well_name]["rates"]
        # [0, row, column]
        iw_lrc = [0] + list(dis5.get_rc_from_node_coordinates(iw[1], iw[2]))
        # Defining list containing stress period data under correct format
        spiw = [iw_lrc + [r] for r in iwr]
        return [iw, iwr, iw_lrc, spiw]

    # Produce well stress period data readable by modflow
    my_wells = [make_well(o) for o in wcd.wells_data]

    spd = np.array([mw[-1] for mw in my_wells])  # Collecting SPD for each well

    np.save(jp(model_ws, "spd"), spd)  # Saves the SPD

    # %% ModflowWel

    well_stress_period_data = {}

    for sp in range(nper):
        well_stress_period_data[sp] = np.array(spd)[:, sp]

    # stress_period_data =
    # {
    # 0: [[    0 ,    50 ,    80 , -2400. ]],
    # 1: [[    0 ,    50 ,    80 , -2400. ]],
    # 2: [[    0 ,    50 ,    80 , -2400. ]]
    # }

    flopy.modflow.ModflowWel(model=model,
                             stress_period_data=well_stress_period_data)

    # %% ModflowBas

    flopy.modflow.ModflowBas(
        model=model,
        ibound=ibound,
        strt=strt,
        ifrefm=True,
        ixsec=False,
        ichflg=False,
        stoper=None,
        hnoflo=-999.99,
        extension="bas",
        unitnumber=None,
        filenames=None,
    )

    # %% ModflowPcg

    flopy.modflow.ModflowPcg(
        model=model,
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
        extension="pcg",
        unitnumber=None,
        filenames=None,
    )

    # %% ModflowOc

    stress_period_data_oc = {}

    for kper in range(nper):  # For each stress period
        for kstp in range(nstp[kper]):  # For each time step per time period
            # MT3DMS needs HEAD
            # MODPATH needs BUDGET
            stress_period_data_oc[(kper, kstp)] = [
                "save head",
                "print head",
                "save budget",
                "print budget",
            ]

    flopy.modflow.ModflowOc(model=model,
                            stress_period_data=stress_period_data_oc,
                            compact=True)

    # %% ModflowLmt
    # Necessary to create a xxx.ftl file to link modflow simulation to mt3dms transport simulation.
    output_file_name = "mt3d_link.ftl"

    flopy.modflow.ModflowLmt(model=model, output_file_name=output_file_name)

    # %% SGEMS
    # Flattening hk_array to plot it
    fl = [item for sublist in hk_array for item in sublist]
    val = []
    # Adding 'nlay' times so all layers get the same conductivity.
    for n in range(nlay):
        val.append(fl)
    val = [item for sublist in val for item in sublist]  # Flattening

    # If the statistical_simulation grid is different from the modflow grid, which might be the case since we
    # would like to refine in some ways the flow mesh, the piece of code below assigns to the flow grid the
    # values of the hk simulations based on the closest distance between cells.
    # Index file location - relates the position of closest cells
    inds_file = jp(grid_dir, "inds.npy")
    # between differently discretized meshes.
    if (
            nrow_d != nrow or ncol_d != ncol
    ):  # if mismatch between nrow and ncol, that is to say, we must copy/paste
        # the new hk array on a new grid.
        if flag_dis == 0:
            # Compute distance matrix between refined and dummy grid.
            dm = distance_matrix(xy_true, xy_dummy)
            inds = [
                np.unravel_index(np.argmin(dm[i], axis=None), dm[i].shape)[0]
                for i in range(dm.shape[0])
            ]
            np.save(inds_file, inds)  # Save index file to avoid re-computing
        else:  # If the inds file exists.
            inds = np.load(inds_file)
        valk = [val[k] for k in inds]  # Contains k values for refined grid
        # Reshape in n layers x n cells in refined grid.
        valkr = np.reshape(valk, (nlay, nrow, ncol))
    else:
        valkr = hk_array[0]

    np.save(jp(model_ws, "hk"), valkr)

    # %% Layer 1 properties

    laytyp = 1  # confined
    layavg = 0
    chani = 0
    layvka = 0
    laywet = 0
    ipakcb = 53
    hdry = -1e30
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

    flopy.modflow.ModflowLpf(
        model=model,
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
        extension="lpf",
    )

    # %% Modflow Run

    model.write_input()

    # Run the model
    model.run_model(silent=True, pause=False, report=True)

    # model.check(level=0)

    # %% Checking flow results

    # Create the headfile and budget file objects
    headobj = bf.HeadFile(jp(model_ws, f"{model_name}.hds"))
    times = headobj.get_times()
    head = headobj.get_data(totim=times[-1])  # Get last data
    headobj.close()

    if (
            head.max() > np.max(top) + 1
    ):  # Quick check - if the maximum computed head is higher than the layer top,
        # it means that an error occurred, and we shouldn't waste time computing the transport on a false solution.
        model = None
    if head.min() == -1e30:
        model = None
    else:
        np.save(jp(model_ws, f"{model_name}_heads.npy"), head)

    return model
