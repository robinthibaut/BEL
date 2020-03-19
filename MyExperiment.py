# %% Import
from multiprocessing import Process
import os
import shutil
from os import listdir
from os.path import join as jp
from os.path import isfile
import skfmm
import mySgems
import numpy as np
import flopy
from scipy.spatial import distance_matrix
from datetime import date
import flopy.utils.binaryfile as bf
from mySgems import MySgems
from matplotlib.patches import Polygon
from TravellingParticles import tsp
import uuid


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


def load_flow_model(nam_file, exe_name, model_ws):
    flow_loader = flopy.modflow.mf.Modflow.load
    return flow_loader(f=nam_file, exe_name=exe_name, model_ws=model_ws)


def load_transport_model(nam_file, modflowmodel, exe_name, model_ws, ftl_file='mt3d_link.ftl', version='mt3d-usgs'):
    transport_loader = flopy.mt3d.Mt3dms.load
    transport_reloaded = transport_loader(f=nam_file, version=version, modflowmodel=modflowmodel,
                                          exe_name=exe_name, model_ws=model_ws)
    transport_reloaded.ftlfilename = ftl_file
    return transport_reloaded


def main():
    # %% Directories
    cwd = os.getcwd()
    exe_loc = jp(cwd, 'exe')

    # %% EXE files directory.
    exe_name_mf = jp(exe_loc, 'mf2005.exe')
    exe_name_mt = jp(exe_loc, 'mt3d.exe')
    exe_name_mp = jp(exe_loc, 'mp7.exe')

    # Main results directory.
    res_dir = uuid.uuid4().hex
    results_dir = jp(cwd, 'results', res_dir)

    # Sgems simulation results directory.
    sgems_simulations_dir = results_dir

    # Grid data directory.
    grid_data_dir = jp(cwd, 'grid')

    # Modflow output directory
    modflow_dir = results_dir

    # Generates the result directory
    dirmaker(results_dir)

    # %% Model name

    model_name = 'whpa'

    # %% Modflow

    model = flopy.modflow.Modflow(modelname=model_name,
                                  namefile_ext='nam',
                                  version='mf2005',
                                  exe_name=exe_name_mf,
                                  structured=True,
                                  listunit=2,
                                  model_ws=modflow_dir,
                                  external_path=None,
                                  verbose=False)

    # %% Model time discretization

    itmuni = 4  # Time units = days
    nper = 3  # Number of stress periods
    perlen = [1, 1 / 12, 100]  # Period lengths
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
    pt = [x_lim * 2 / 3, y_lim / 2]  # Point around which refinement will occur
    # r_params = np.array([[8, 100], [6, 80], [4, 50], [3, 30], [2, 20], [1, 10]])
    r_params = [[10, 10]]
    delc = np.ones(ncol) * dx  # Size of each cell in x-dimension
    delr = np.ones(nrow) * dy  # Size of each cell in y-dimension

    # Refinement:
    for p in r_params:
        delc = refine_axis(delc, pt[0], p[1], p[0], dx, x_lim)
        delr = refine_axis(delr, pt[1], p[1], p[0], dy, y_lim)

    ncol = len(delc)
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

    min_cell_dim = 10
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

    for yc in ncd0[0]:
        for xc in ncd0[1]:
            xyz_dummy.append([xc, yc])

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
                                    unitnumber=None,
                                    filenames=None,
                                    xul=None,
                                    yul=None,
                                    rotation=0.0,
                                    proj4_str=None,
                                    start_datetime=start_datetime)

    ncd1 = dis5.get_node_coordinates()  # Get y, x, and z cell centroids.
    xyz_true = []
    for yc in ncd1[0]:
        for xc in ncd1[1]:
            xyz_true.append([xc, yc])

    def get_node_id(dis, x, y):
        node_rc = dis.get_rc_from_node_coordinates(x, y)  # Get node number of the wel location
        node = dis.get_node((0,) + node_rc)[0]
        return node

    # (layer, y, x)
    pumping_well_lrc = [0, pt[0], pt[1]]  # Pumping well location in model coordinates
    pumping_well_node = get_node_id(dis5, pt[0], pt[1])  # Pumping well node number in refined grid.

    def make_well(wel_info):
        """
        Produce well stress period data readable by modflow
        :param wel_info: [ r, c, [rate sp #0, ..., rate sp# n] ]
        :return:
        """
        # TODO: fix this function
        iw = [0, wel_info[0], wel_info[1]]  # Injection wel location - layer row column
        iwr = wel_info[2]  # Well rate for the defined time periods
        iw_lrc = [0] + list(dis5.get_rc_from_node_coordinates(iw[1], iw[2]))
        spiw = [iw_lrc + [r] for r in iwr]  # Defining list containing stress period data under correct format
        return [iw, iwr, iw_lrc, spiw]

    # wells_data = [[pumping_well_lrc[1], pumping_well_lrc[2], [-1000, -1000, -1000]],  # PW followed by IW
    #               [pumping_well_lrc[1] - 50, pumping_well_lrc[2] - 0, [0, 24, 0]],
    #               [pumping_well_lrc[1] - 0, pumping_well_lrc[2] + 40, [0, 24, 0]],
    #               [pumping_well_lrc[1] + 35, pumping_well_lrc[2] + 0, [0, 24, 0]],
    #               [pumping_well_lrc[1] + 0, pumping_well_lrc[2] - 50, [0, 24, 0]]]

    wells_data = [[pumping_well_lrc[1], pumping_well_lrc[2], [-1000, -1000, -1000]],
                  [pumping_well_lrc[1] - 50, pumping_well_lrc[2] - 0, [0, 24, 0]]]

    my_wels = [make_well(o) for o in wells_data]

    spd = [mw[-1] for mw in my_wels]
    spd = np.array(spd)

    # %% ModflowWel

    wel_stress_period_data = {}

    for sp in range(nper):
        wel_stress_period_data[sp] = np.array(spd)[:, sp]

    # stress_period_data = {0: [[    0 ,    50 ,    80 , -2400. ]], 1: [[    0 ,    50 ,    80 , -2400. ]], 2: [[    0 ,
    # 50 ,    80 , -2400. ]]}

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

    for kper in range(nper):
        for kstp in range(nstp[kper]):
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

    output_file_name = 'mt3d_link.ftl'

    lmt = flopy.modflow.ModflowLmt(model=model,
                                   output_file_name=output_file_name)

    # %% SGEMS

    op = 'hk'  # Simulations output name

    sgems = MySgems()

    nr = 1  # Number of realizations.

    wells_nodes_sgems = [get_node_id(dis_sgems, w[0], w[1]) for w in wells_data]

    # Hard data node
    # fixed_nodes = [[pwnode_sg, 2], [iw1node_sg, 1], [iw2node_sg, 1.5], [iw3node_sg, 0.7], [iw4node_sg, 1.2]]
    fixed_nodes = [[w, 1 + np.random.rand()] for w in wells_nodes_sgems]

    sgrid = [dis_sgems.ncol,
             dis_sgems.nrow,
             dis_sgems.nlay,
             dis_sgems.delc.array[0],
             dis_sgems.delr.array[0],
             dis_sgems.delr.array[0],
             xo, yo, zo]  # Grid information

    seg = [50, 50, 50, 0, 0, 0]  # Search ellipsoid geometry

    sgems.sims(op_folder=sgems_simulations_dir.replace('\\', '//'),
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

    opl = jp(sgems_simulations_dir, op + '.grid')  # Output file location.

    hk = mySgems.so(opl)  # Grid information directly derived from the output file.

    k_mean = np.random.uniform(1.4, 2)  # Hydraulic conductivity mean between x and y in m/d.

    # k_std = np.random.uniform(0.2, 0.55)  # Hydraulic conductivity variance.
    k_std = 0.4

    hkp = np.copy(hk)

    HK = [mySgems.o2k2(h, k_mean, k_std) for h in hkp]

    # Setting the hydraulic conductivity matrix.
    HK = [np.reshape(h, (nlay, dis_sgems.nrow, dis_sgems.ncol)) for h in HK]

    HK = [np.fliplr(HK[h]) for h in range(0, nr, 1)]  # Flip to correspond to sgems grid! HK is now a list of the arrays
    # of conductivities for each realization.

    with open(jp(sgems_simulations_dir, 'hk_t.dat'), 'w') as hkf:
        for s in range(len(HK)):
            hkf.write(str(nr) + '\n')
            hkf.write(str(dis_sgems.ncol * dis_sgems.nrow) + '\n')
            hkf.write('simulation ' + str(s + 1) + '\n')
            hf = HK[s].flatten()
            [hkf.write(str(h) + '\n') for h in hf]
        hkf.close()

    # Flattening HK to plot it
    fl = [item for sublist in HK[0] for item in sublist]
    fl2 = [item for sublist in fl for item in sublist]
    val = []
    for n in range(nlay):  # Adding 'nlay' times so all layers get the same conductivity.
        val.append(fl2)
    val = [item for sublist in val for item in sublist]

    np.save(jp(results_dir, 'hk'), HK[0])

    # If the sgems grid is different from the modflow grid, which might be the case since we would like to refine
    # in some ways the flow mesh, the piece of code below assigns to the flow grid the values of the hk simulations
    # based on the closest distance between cells.
    # TODO: This can take a long time, I should compute this before launching the n simulations.
    if nrow_d != nrow or ncol_d != ncol:
        dm = distance_matrix(xyz_true, xyz_dummy)
        inds = [np.unravel_index(np.argmin(dm[i], axis=None), dm[i].shape)[0] for i in range(dm.shape[0])]
        valk = [val[k] for k in inds]  # Contains k values for refined grid
        valkr = np.reshape(valk, (nlay, nrow, ncol))  # Reshape in n layers x n cells in refined grid.
    else:
        valkr = HK[0]

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

    hk = valkr  # Hydraulic conductivity

    hani = 1.0
    vka = hk / 10
    ss = 10e-4
    sy = 0.25
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
                                   extension='lpf',
                                   unitnumber=None,
                                   filenames=None)

    # lpf.check(level=0)

    # %% Modflow Run

    model.write_input()

    # Run the model
    model.run_model(silent=True,
                    pause=False,
                    report=True)
    # if not success:
    #     raise Exception('MODFLOW did not terminate normally.')

    # model.check(level=0)

    # %% Checking flow results

    headobj = bf.HeadFile(jp(modflow_dir, model_name + '.hds'))  # Create the headfile and budget file objects
    times = headobj.get_times()
    # cbb = bf.CellBudgetFile(jp(model_ws, model_name + '.cbc'))
    mytimes = [times[-1]]  # Taking last time step
    #
    head = headobj.get_data(totim=times[-1])
    headobj.close()

    if head.max() > top + 1:  # Quick check - if the maximum computed head is higher than the layer top, it means
        # that an error occured, and we shouldn't waste time computing the transport on a false solution.
        # TODO: optimize this
        exit()

    # TODO: Facilitate results visualization and plot results

    # %% Mt3dms

    version = 'mt3d-usgs'
    namefile_ext = 'mtnam'
    ftlfilename = output_file_name

    mt = flopy.mt3d.Mt3dms(modflowmodel=model,
                           ftlfilename=ftlfilename,
                           modelname=model_name,
                           model_ws=modflow_dir,
                           version=version,
                           namefile_ext=namefile_ext,
                           exe_name=exe_name_mt)

    # mt.check(level=0)

    # %% Mt3dBtn

    # I will have to define an active zone because otherwise it takes a long time.
    # Start from the coordinates of the cells and use np.where

    # Defining mt3d active zone - I take the maximum distance from an IW to the PW and add 20m.
    ext_a = np.abs(np.array(wells_data)[:, :2] - np.array(pumping_well_lrc)[1:]).max() + 20
    # Extent in meters around the well
    x_inf = pumping_well_lrc[1] - ext_a
    x_sup = pumping_well_lrc[1] + ext_a
    y_inf = pumping_well_lrc[2] - ext_a
    y_sup = pumping_well_lrc[2] + ext_a

    cdx = np.reshape(xyz_true, (nlay, nrow, ncol, 2))  # Coordinates of centroids

    icbund = np.zeros((nlay, nrow, ncol))  # zero=array with grid shape
    ind_a = np.where((cdx[:, :, :, 0] > x_inf) & (cdx[:, :, :, 0] < x_sup) &
                     (cdx[:, :, :, 1] > y_inf) & (cdx[:, :, :, 1] < y_sup))
    icbund[ind_a] = 1  # Coordinates around wel are active

    # model_map(grid1, vals=np.reshape(icbund, nrow*ncol), log=0)
    # # plt.plot(pw[0], pw[1], 'ko')
    # # plt.title('Transport active zone')
    # plt.show()

    MFStyleArr = False
    DRYCell = False
    Legacy99Stor = False
    FTLPrint = False
    NoWetDryPrint = False
    OmitDryBud = False
    AltWTSorb = False
    ncomp = len(my_wels) - 1  # The number of components
    mcomp = ncomp
    tunit = 'D'
    lunit = 'M'
    munit = 'KG'
    prsity = sy  # 0.25
    # icbund = 1
    sconc = 0
    # sconc2 = 0
    # sconc3 = 0
    cinact = 1e+30
    thkmin = 0.01
    ifmtcn = 0
    ifmtnp = 0
    ifmtrf = 0
    ifmtdp = 0
    savucn = True
    nprs = 0
    timprs = None
    obs_lxy = [pumping_well_lrc[0], pumping_well_lrc[1], pumping_well_lrc[2]]
    obs = list(map(int, [0] + list(dis5.get_rc_from_node_coordinates(obs_lxy[1], obs_lxy[2]))))
    obs = [tuple(obs)]
    nprobs = 1
    chkmas = True
    nprmas = 1
    perlen = None
    nstp = None
    tsmult = None
    ssflag = None
    dt0 = 0
    mxstrn = 50000
    ttsmult = 1.0
    ttsmax = 0
    species_names = ['c{}'.format(c + 1) for c in range(ncomp)]
    extension = 'btn'
    unitnumber = None
    filenames = None

    btn = flopy.mt3d.mtbtn.Mt3dBtn(model=mt,
                                   MFStyleArr=MFStyleArr,
                                   DRYCell=DRYCell,
                                   Legacy99Stor=Legacy99Stor,
                                   FTLPrint=FTLPrint,
                                   NoWetDryPrint=NoWetDryPrint,
                                   OmitDryBud=OmitDryBud,
                                   AltWTSorb=AltWTSorb,
                                   ncomp=ncomp,
                                   mcomp=mcomp,
                                   tunit=tunit,
                                   lunit=lunit,
                                   munit=munit,
                                   prsity=prsity,
                                   icbund=icbund,
                                   sconc=sconc,
                                   # sconc2=sconc2,
                                   # sconc3=sconc3,
                                   cinact=cinact,
                                   thkmin=thkmin,
                                   ifmtcn=ifmtcn,
                                   ifmtnp=ifmtnp,
                                   ifmtrf=ifmtrf,
                                   ifmtdp=ifmtdp,
                                   savucn=savucn,
                                   nprs=nprs,
                                   timprs=timprs,
                                   obs=obs,
                                   nprobs=nprobs,
                                   chkmas=chkmas,
                                   nprmas=nprmas,
                                   perlen=perlen,
                                   nstp=nstp,
                                   tsmult=tsmult,
                                   ssflag=ssflag,
                                   dt0=dt0,
                                   mxstrn=mxstrn,
                                   ttsmult=ttsmult,
                                   ttsmax=ttsmax,
                                   species_names=species_names,
                                   extension=extension,
                                   unitnumber=unitnumber,
                                   filenames=filenames)

    # %% Mt3dAdv

    mixelm = 3
    percel = 0.75
    mxpart = 800000
    nadvfd = 1
    itrack = 3
    wd = 0.5
    dceps = 1e-5
    nplane = 2
    npl = 0
    nph = 40
    npmin = 3
    npmax = 80
    nlsink = 0
    npsink = 15
    dchmoc = 0.0001
    extension = 'adv'
    unitnumber = None
    filenames = None

    adv = flopy.mt3d.Mt3dAdv(model=mt,
                             mixelm=mixelm,
                             percel=percel,
                             mxpart=mxpart,
                             nadvfd=nadvfd,
                             itrack=itrack,
                             wd=wd,
                             dceps=dceps,
                             nplane=nplane,
                             npl=npl,
                             nph=nph,
                             npmin=npmin,
                             npmax=npmax,
                             nlsink=nlsink,
                             npsink=npsink,
                             dchmoc=dchmoc,
                             extension=extension,
                             unitnumber=unitnumber,
                             filenames=filenames)

    # %% Mt3dDsp

    al = 3
    trpt = 0.1
    trpv = 0.01
    dmcoef = 1e-9
    # dmcoef2 = 1e-9
    # dmcoef3 = 1e-9
    extension = 'dsp'
    multiDiff = False
    unitnumber = None
    filenames = None

    dsp = flopy.mt3d.Mt3dDsp(model=mt,
                             al=al,
                             trpt=trpt,
                             trpv=trpv,
                             dmcoef=dmcoef,
                             # dmcoef2=dmcoef2,
                             # dmcoef3=dmcoef3,
                             extension=extension,
                             multiDiff=multiDiff,
                             unitnumber=unitnumber,
                             filenames=filenames)

    # %% Mt3dSsm

    crch = None
    cevt = None
    mxss = None  # If problem, set to 10000
    dtype = None
    extension = 'ssm'
    unitnumber = None
    filenames = None

    # [(‘k’, np.int), (“i”, np.int), (“j”, np.int), (“css”, np.float32),
    # (“itype”, np.int), ((cssms(n), np.float), n=1, ncomp)]

    stress_period_data = {}

    # layer, row, column, concentration specie 1 (dummy value), type, concentration specie 1,
    # concentration specie 2, concentration specie 3

    def spd_maker(l_r_c, c):
        """

        :param l_r_c: L R C of IW
        :param c: Array containing injected concentration for each stress period
        :return:
        """
        # FIXME: In case of single-component simulation
        # DONE
        nsp = len(c[0])  # Number of stress periods
        nc = len(l_r_c)  # Number of components
        spdt = {}

        for i in range(nsp):
            spdt_i = []
            for j in range(nc):
                cc = np.zeros(nc)
                cc[j] = c[j][i]
                c0 = 0
                if j == 0:
                    c0 = cc[0]
                if ncomp > 1:
                    t = tuple(np.concatenate((np.array([l_r_c[j][0], l_r_c[j][1], l_r_c[j][2], c0, 2]), cc), axis=0))
                else:
                    t = tuple([l_r_c[j][0], l_r_c[j][1], l_r_c[j][2], c0, 2])
                spdt_i.append(t)
            spdt[i] = spdt_i
        return spdt

    iw_lrc = np.array(my_wels)[1:, 2]  # R L C of each IW in the refined flow grid. The first element
    # correspond to the PW and is not taken.
    iw_pr = np.array([[0, 1.5, 0] for c in range(ncomp)])  # Injection rate for each time period. We decide to assign
    # the same for each IW.
    # The units are KG/D

    stress_period_data = spd_maker(iw_lrc, iw_pr)

    # Template
    # stress_period_data[0] = [(iw1_lrc[0], iw1_lrc[1], iw1_lrc[2], 0, 2, 0, 0, 0, 0),
    #                          (iw2_lrc[0], iw2_lrc[1], iw2_lrc[2], 0, 2, 0, 0, 0, 0),
    #                          (iw3_lrc[0], iw3_lrc[1], iw3_lrc[2], 0, 2, 0, 0, 0, 0),
    #                          (iw4_lrc[0], iw4_lrc[1], iw4_lrc[2], 0, 2, 0, 0, 0, 0)]
    #
    # stress_period_data[1] = [(iw1_lrc[0], iw1_lrc[1], iw1_lrc[2], 1.500, 2, 1.500, 0, 0, 0),
    #                          (iw2_lrc[0], iw2_lrc[1], iw2_lrc[2], 0, 2, 0, 1.500, 0, 0),
    #                          (iw3_lrc[0], iw3_lrc[1], iw3_lrc[2], 0, 2, 0, 0, 1.500, 0),
    #                          (iw4_lrc[0], iw4_lrc[1], iw4_lrc[2], 0, 2, 0, 0, 0, 1.500)]
    #
    # stress_period_data[2] = [(iw1_lrc[0], iw1_lrc[1], iw1_lrc[2], 0, 2, 0, 0, 0, 0),
    #                          (iw2_lrc[0], iw2_lrc[1], iw2_lrc[2], 0, 2, 0, 0, 0, 0),
    #                          (iw3_lrc[0], iw3_lrc[1], iw3_lrc[2], 0, 2, 0, 0, 0, 0),
    #                          (iw4_lrc[0], iw4_lrc[1], iw4_lrc[2], 0, 2, 0, 0, 0, 0)]

    # stress_period_data[0] = [(iw1[0], iw1[1], iw1[2], 0, 2)]
    # stress_period_data[1] = [(iw1[0], iw1[1], iw1[2], 1500, 2)]
    # stress_period_data[2] = [(iw1[0], iw1[1], iw1[2], 0, 2)]

    ssm = flopy.mt3d.Mt3dSsm(model=mt,
                             crch=crch,
                             cevt=cevt,
                             mxss=mxss,
                             stress_period_data=stress_period_data,
                             dtype=dtype,
                             extension=extension,
                             unitnumber=unitnumber,
                             filenames=filenames)

    # %% Mt3dGcg

    mxiter = 1
    iter1 = 50
    isolve = 3
    ncrs = 1
    accl = 1
    cclose = 1e-5
    iprgcg = 0
    extension = 'gcg'
    unitnumber = None
    filenames = None

    gcg = flopy.mt3d.Mt3dGcg(model=mt,
                             mxiter=mxiter,
                             iter1=iter1,
                             isolve=isolve,
                             ncrs=ncrs,
                             accl=accl,
                             cclose=cclose,
                             iprgcg=iprgcg,
                             extension=extension,
                             unitnumber=unitnumber,
                             filenames=filenames)

    # %% MT3D run

    mt.write_input()

    mt.run_model(silent=True,
                 pause=False,
                 report=True)

    # %% Checking transport results

    #  Loading observations files.

    only_files = [f for f in listdir(modflow_dir) if isfile(jp(modflow_dir, f))]  # Listing all files in results folder

    obs_files = [jp(modflow_dir, x) for x in only_files if '.OBS' in x]  # Selects OBS files

    transport_model = flopy.mt3d.Mt3dms.load(
        jp(modflow_dir, model_name + '.{}'.format(namefile_ext)))  # Loads the transport model

    observations = [transport_model.load_obs(of) for of in obs_files]  # Loading observations results
    fields = observations[0].dtype.names

    hey = np.array(
        [list(zip(observations[i][fields[1]], observations[i][fields[2]])) for i in range(len(observations))])
    # ny = np.array([savgol_filter(ob[:, 1], 51, 3) for ob in hey])  # smoothed res
    # TODO: Adding IW coordinates when exporting results in binary
    np.save(jp(results_dir, 'bkt'), hey)  # saves raw curves

    # %% MODPATH

    mpfname = model_name + '_mp'
    # model_ws_mp = jp(model_ws, 'modpath')
    #
    # if not os.path.isdir(model_ws_mp):
    #     os.mkdir(model_ws_mp)

    wn = pumping_well_node  # Wel node

    # Create particles

    # %% Particle group

    drape = 0
    columncelldivisions = 15
    rowcelldivisions = 15
    layercelldivisions = 1

    sd = flopy.modpath.CellDataType(drape=drape,
                                    columncelldivisions=columncelldivisions,
                                    rowcelldivisions=rowcelldivisions,
                                    layercelldivisions=layercelldivisions)

    p = flopy.modpath.mp7particledata.NodeParticleData(subdivisiondata=sd,
                                                       nodes=wn)

    pg = flopy.modpath.ParticleGroupNodeTemplate(particlegroupname='PG',
                                                 particledata=p,
                                                 filename='pg.sloc')

    particlegroups = [pg]

    # %% Default iface for MODFLOW-2005
    defaultiface = {'RECHARGE': 6, 'ET': 6}

    # %% Create modpath files

    mp = flopy.modpath.Modpath7(modelname=mpfname,
                                flowmodel=model,
                                exe_name=exe_name_mp,
                                model_ws=modflow_dir)

    porosity_mp = 0.2
    mpbas = flopy.modpath.Modpath7Bas(mp,
                                      porosity=porosity_mp,
                                      defaultiface=defaultiface)

    simulationtype = 'combined'
    trackingdirection = 'backward'
    weaksinkoption = 'pass_through'
    weaksourceoption = 'pass_through'
    budgetoutputoption = 'summary'
    budgetcellnumbers = None
    traceparticledata = None
    referencetime = None
    stoptimeoption = 'specified'
    stoptime = 30
    timepointdata = None
    zonedataoption = 'off'
    zones = None

    # %% Run modpath simulation

    mpsim = flopy.modpath.Modpath7Sim(mp,
                                      simulationtype=simulationtype,
                                      trackingdirection=trackingdirection,
                                      weaksinkoption=weaksinkoption,
                                      weaksourceoption=weaksourceoption,
                                      budgetoutputoption=budgetoutputoption,
                                      budgetcellnumbers=budgetcellnumbers,
                                      traceparticledata=traceparticledata,
                                      referencetime=referencetime,
                                      stoptimeoption=stoptimeoption,
                                      stoptime=stoptime,
                                      timepointdata=timepointdata,
                                      zonedataoption=zonedataoption, zones=zones,
                                      particlegroups=particlegroups)

    # Write modpath datasets
    mp.write_input()

    # Run modpath
    mp.run_model(silent=True)

    # %% Loading MODPATH results
    # Load backward tracking path lines

    # fpth = jp(model_ws, mpfname + '.mppth')
    # gp = flopy.utils.PathlineFile(fpth)
    # pwb = gp.get_destination_pathline_data(dest_cells=wn)

    # Load backward tracking endpoints
    modpath_files = jp(modflow_dir, mpfname + '.mpend')
    e = flopy.utils.EndpointFile(modpath_files)
    ewb = e.get_destination_endpoint_data(dest_cells=wn, source=True)
    # In my implementation I'm only interested in x y locations.
    xep = ewb.x  # Endpoints x-locations (m)
    yep = ewb.y  # Endpoints y-locations (m)
    pzone_xy = np.array(list(zip(xep, yep)))

    # %% Signed distance

    pz = np.copy(pzone_xy)  # x-y coordinates endpoints particles
    delineation = tsp(pz)  # indices of the vertices of the final protection zone using TSP algorithm
    pzs = pz[delineation]  # x-y coordinates protection zone
    np.save(jp(results_dir, 'pz'), pzs)

    # Points locations density
    # from scipy.stats import gaussian_kde
    # x = pz[:, 0]
    # y = pz[:, 1]
    # Calculate the point density
    # xy = np.vstack([x, y])
    # z = gaussian_kde(xy)(xy)
    # fig, ax = plt.subplots()
    # ax.scatter(x, y, c=z, s=100, edgecolor='')
    # plt.show()

    # Polygon approach - best

    poly = Polygon(pzs, True)

    grf = 1  # Cell size
    x_lim = 1500
    y_lim = 1000
    nrow = int(y_lim / grf)
    ncol = int(x_lim / grf)
    phi = np.ones((nrow, ncol)) * -1
    xys = np.dstack((np.flip((np.indices(phi.shape) + 1), 0) * grf - grf / 2))  # Getting centroids

    xys = xys.reshape((nrow * ncol, 2))
    ind = np.nonzero(poly.contains_points(xys))[0]  # Checks which points are enclosed by polygon.
    phi = phi.reshape((nrow * ncol))
    phi[ind] = 1  # Points inside the WHPA are assigned a value of 1, and 0 for those outside
    phi = phi.reshape((nrow, ncol))

    sd = skfmm.distance(phi, dx=grf)  # Signed distance computation
    np.save(jp(results_dir, 'sd'), sd)

    # Deletes everything excepts final results
    # TODO: in order to implement new tracing experiments,
    #  I should check which files mt3dms needs to work with and not deleting them
    # for the_file in os.listdir(results_dir):
    #     if not the_file.endswith('.npy') and not the_file.endswith('.py'):
    #         file_path = os.path.join(results_dir, the_file)
    #         try:
    #             if os.path.isfile(file_path):
    #                 os.unlink(file_path)
    #             elif os.path.isdir(file_path):
    #                 shutil.rmtree(file_path)
    #         except Exception as e:
    #             print(e)
    shutil.copy(__file__, jp(results_dir, 'copied_script.py'))  # Copies this script to the result file


if __name__ == "__main__":
    # jobs = []
    # for j in range(50):
    #     for i in range(3):  # Can run max 3 instances of mt3dms at once on this computer
    #         process = Process(target=main)
    #         jobs.append(process)
    #         process.start()
    #     process.join()
    #     process.close()
    main()
