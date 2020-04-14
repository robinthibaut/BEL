from os import listdir
from os.path import isfile
from os.path import join as jp

import flopy
import numpy as np


def transport(modflowmodel, exe_name):
    """

    :param modflowmodel: flopy modflow model object
    :param exe_name: Path to executable file
    :return:
    """

    version = 'mt3d-usgs'
    namefile_ext = 'mtnam'
    ftlfilename = 'mt3d_link.ftl'
    # Extract working directory and name from modflow object
    model_ws = modflowmodel.model_ws
    modelname = modflowmodel.name

    # Initiate Mt3dms object
    mt = flopy.mt3d.Mt3dms(modflowmodel=modflowmodel,
                           ftlfilename=ftlfilename,
                           modelname=modelname,
                           model_ws=model_ws,
                           version=version,
                           namefile_ext=namefile_ext,
                           exe_name=exe_name)

    # Extract discretization info from modflow object
    dis = modflowmodel.dis  # DIS package
    nlay = modflowmodel.nlay
    nrow = modflowmodel.nrow
    ncol = modflowmodel.ncol
    # yxz_grid is an array of the coordinates of each node:
    # [[coordinates Y], [coordinates X], [coordinates Z]]
    yxz_grid = dis.get_node_coordinates()
    xy_true = []  # Convert to 2D array
    for yc in yxz_grid[0]:
        for xc in yxz_grid[1]:
            xy_true.append([xc, yc])
    xy_nodes_2d = np.reshape(xy_true, (nlay * nrow * ncol, 2))  # Flattens xy to correspond to node numbering

    wells_data = np.load(jp(model_ws, 'spd.npy'))  # Loads well stress period data
    wells_number = wells_data.shape[0]  # Total number of wells
    pumping_well_data = wells_data[0]  # Pumping well in first
    pw_lrc = pumping_well_data[0][:3]  # PW layer row column
    pw_node = int(dis.get_node(pw_lrc)[0])  # PW node number
    xy_pumping_well = xy_nodes_2d[pw_node]  # Get PW x, y coordinates (meters) from well node number

    injection_well_data = wells_data[1:]
    iw_nodes = [int(dis.get_node(w[0][:3])[0]) for w in injection_well_data]
    xy_injection_wells = [xy_nodes_2d[iwn] for iwn in iw_nodes]
    # TODO: There seems to be a problem with their get_lrc() method - issue submitted

    lpf = modflowmodel.lpf  # LPF package

    # %% Mt3dBtn

    # I will have to define an active zone because otherwise it takes a long time.
    # Start from the coordinates of the cells and use np.where
    # Defining mt3d active zone - I take the maximum distance from an IW to the PW and add 20m.
    # TODO: change this and use signed_distance module to create a polygon as an active zone
    from bel.hydro.whpa.travelling_particles import tsp
    from bel.processing.signed_distance import SignedDistance
    sdm = SignedDistance()
    sdm.xys = xy_true
    sdm.nrow = nrow
    sdm.ncol = ncol
    xyw_scaled = (xy_injection_wells - xy_pumping_well)*1.5 + xy_pumping_well
    poly_deli = tsp(xyw_scaled)
    poly_xyw = xyw_scaled[poly_deli]
    icbund = sdm.matrix_poly_bin(poly_xyw, outside=0, inside=1).reshape(nlay, nrow, ncol)
    # Need function to copy/paste on grids
    # ext_a = np.abs(xy_injection_wells - xy_pumping_well).max() + 20
    # Extent in meters around the well
    # x_inf = xy_pumping_well[0] - ext_a
    # x_sup = xy_pumping_well[0] + ext_a
    # y_inf = xy_pumping_well[1] - ext_a
    # y_sup = xy_pumping_well[1] + ext_a
    #
    # cdx = np.reshape(xy_true, (nlay, nrow, ncol, 2))  # Coordinates of centroids of refined grid
    #
    # icbund = np.zeros((nlay, nrow, ncol))  # zero=array with grid shape
    # ind_a = np.where((cdx[:, :, :, 0] > x_inf) & (cdx[:, :, :, 0] < x_sup) &
    #                  (cdx[:, :, :, 1] > y_inf) & (cdx[:, :, :, 1] < y_sup))
    # icbund[ind_a] = 1  # Coordinates around wel are active
    from diavatly import model_map
    import matplotlib.pyplot as plt
    from bel.toolbox.mesh_ops import MeshOps
    mo = MeshOps()
    grid1 = mo.blocks_from_rc(dis.delc, dis.delr)
    model_map(grid1, vals=np.reshape(icbund, nrow*ncol), log=0)
    plt.plot(xy_pumping_well[0], xy_pumping_well[1], 'ko', markersize=200)
    plt.show()


    MFStyleArr = False
    DRYCell = False
    Legacy99Stor = False
    FTLPrint = False
    NoWetDryPrint = False
    OmitDryBud = False
    AltWTSorb = False
    ncomp = wells_number - 1  # The number of components = number of injecting wells
    mcomp = ncomp
    tunit = 'D'
    lunit = 'M'
    munit = 'KG'
    prsity = lpf.sy.array  # Porosity loaded from the LPF package = sy
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
    savucn = False
    nprs = 0
    timprs = None
    obs = [tuple(pw_lrc)]  # Observation point = PW location (layer row column)
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

    # noinspection PyTypeChecker
    flopy.mt3d.mtbtn.Mt3dBtn(model=mt,
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

    # noinspection PyTypeChecker
    flopy.mt3d.Mt3dAdv(model=mt,
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

    # noinspection PyTypeChecker
    flopy.mt3d.Mt3dDsp(model=mt,
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

    # Template:
    # [(‘k’, np.int), (“i”, np.int), (“j”, np.int), (“css”, np.float32),
    # (“itype”, np.int), ((cssms(n), np.float), n=1, ncomp)]

    # layer, row, column, concentration specie 1 (dummy value), type, concentration specie 1,
    # concentration specie 2, concentration specie 3

    def spd_maker(l_r_c, c):
        """
        Given the location (layer, row, column) and pumping rate for each stress period, for >= 1 wells,
        creates a stress period data array that mt3dms needs.
        :param l_r_c: L R C of injection well
        :param c: Array containing injected concentration for each stress period
        :return:
        """
        nsp = len(c[0])  # Number of stress periods
        nc = len(l_r_c)  # Number of components
        spdt = {}

        for i in range(nsp):  # for each stress period
            spdt_i = []  # empty spd array
            for j in range(nc):  # for each well
                cc = np.zeros(nc)  # (0, 0...)
                cc[j] = c[j][i]  # each well -> pumping rate #j at stress period #i
                c0 = 0  # concentration
                if j == 0:  # if first well, need a dummy value
                    c0 = cc[0]  # concentration #0 -> concentration well #0
                if ncomp > 1:  # if more than 1 injecting wells (assuming 1 different comp per iw)
                    t = tuple(np.concatenate((np.array([l_r_c[j][0], l_r_c[j][1], l_r_c[j][2], c0, 2]), cc), axis=0))
                else:
                    t = tuple([l_r_c[j][0], l_r_c[j][1], l_r_c[j][2], c0, 2])
                spdt_i.append(t)  # Each well spd gets added to the empty spd array
            spdt[i] = spdt_i
        return spdt

    iw_lrc = injection_well_data[:, 0, :3]  # R L C of each IW in the refined flow grid
    iw_pr = np.array([[0, 1.5, 0] for _ in range(ncomp)])  # Injection rate for each time period. We decide to assign
    # the same for each IW.
    # The units are KG/D

    stress_period_data = spd_maker(iw_lrc, iw_pr)

    # Template:
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

    # noinspection PyTypeChecker
    flopy.mt3d.Mt3dSsm(model=mt,
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

    # noinspection PyTypeChecker
    flopy.mt3d.Mt3dGcg(model=mt,
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

    # Export
    only_files = [f for f in listdir(model_ws) if isfile(jp(model_ws, f))]  # Listing all files in results folder
    obs_files = [jp(model_ws, x) for x in only_files if '.OBS' in x]  # Selects OBS files
    observations = [mt.load_obs(of) for of in obs_files]  # Loading observations results
    fields = observations[0].dtype.names
    hey = np.array(
        [
            list(zip(observations[i][fields[1]], observations[i][fields[2]])) for i in range(len(observations))
        ]
    )

    np.save(jp(model_ws, 'bkt'), hey)  # saves raw curves

    return mt


def transport_export(model_ws, transport_model_name):
    # Checking transport results

    #  Loading observations files.

    only_files = [f for f in listdir(model_ws) if isfile(jp(model_ws, f))]  # Listing all files in results folder

    obs_files = [jp(model_ws, x) for x in only_files if '.OBS' in x]  # Selects OBS files

    transport_model = flopy.mt3d.Mt3dms.load(jp(model_ws, transport_model_name))  # Loads the transport model

    observations = [transport_model.load_obs(of) for of in obs_files]  # Loading observations results
    fields = observations[0].dtype.names

    hey = np.array(
        [list(zip(observations[i][fields[1]], observations[i][fields[2]])) for i in range(len(observations))])

    # Saving injecting wells location in  a file
    # ssm = transport_model.ssm
    # df = ssm.stress_period_data.df.values
    # iwn = df[:, 1:3]  # nodes
    # grid = transport_model.modelgrid
    # iw_xy = [np.mean(grid.get_cell_vertices(int(i[0]), int(i[1])), axis=0) for i in iwn]
    #
    # np.savetxt('wells.xy', iw_xy)  # Saves injecting wells location

    np.save(jp(model_ws, 'bkt'), hey)  # saves raw curves
