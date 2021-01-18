#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
import re
from os import listdir
from os.path import isfile
from os.path import join as jp

import flopy
import numpy as np


def transport(modflowmodel,
              exe_name: str,
              grid_dir: str,
              save_ucn: bool = False):
    """
    :param modflowmodel: flopy modflow model object.
    :param exe_name: Path to executable file.
    :param grid_dir: Directory containing discretization information.
    :param save_ucn: Flag to save UCN files.
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
    # yxz_grid is an array of the coordinates of each node:
    # [[coordinates Y], [coordinates X], [coordinates Z]]
    yxz_grid = dis.get_node_coordinates()
    xy_true = []  # Convert to 2D array
    for yc in yxz_grid[0]:
        for xc in yxz_grid[1]:
            xy_true.append([xc, yc])

    wells_data = np.load(jp(model_ws, 'spd.npy'))  # Loads well stress period data
    wells_number = wells_data.shape[0]  # Total number of wells
    pumping_well_data = wells_data[0]  # Pumping well in first
    pw_lrc = pumping_well_data[0][:3]  # PW layer row column

    injection_well_data = wells_data[1:]

    lpf = modflowmodel.lpf  # LPF package

    # %% Mt3dBtn

    # Load previously defined active zone
    mt_icbund_file = jp(grid_dir, 'mt3d_icbund.npy')
    icbund = np.load(mt_icbund_file)
    times = np.cumsum(dis.perlen.array)
    tmstp = np.linspace(times[0], times[-1], 100)

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
    sconc = 0  # Starting concentration
    cinact = 1e+30
    thkmin = 0.01
    ifmtcn = 0
    ifmtnp = 0
    ifmtrf = 0
    ifmtdp = 0
    savucn = save_ucn  # Save concentration array or not
    nprs = len(tmstp)  # A flag indicating (i) the frequency of the output and (ii) whether the output frequency is
    # specified in terms of total elapsed simulation time or the transport step number. If nprs > 0 results will be
    # saved at the times as specified in timprs; if nprs = 0, results will not be saved except at the end of
    # simulation; if NPRS 0, simulation results will be saved whenever the number of transport steps is an even
    # multiple of nprs. ( default is 0).
    timprs = tmstp  # The total elapsed time at which the simulation results are saved. The number of entries in
    # timprs must equal nprs. (default is None).
    obs = [tuple(pw_lrc)]  # Observation point = PW location (layer row column)
    nprobs = 1  # An integer indicating how frequently the concentration at the specified observation points should
    # be saved. (default is 1).
    chkmas = True
    nprmas = 1
    perlen = None
    nstp = None
    tsmult = None
    ssflag = None
    dt0 = 0
    mxstrn = 50000
    ttsmult = 1.0  # The multiplier for successive transport steps within a flow time-step if the GCG solver is used
    # and the solution option for the advection term is the standard finite-difference method. (default is 1.0)
    ttsmax = 0
    species_names = [f'c{c+1}' for c in range(ncomp)]
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

    # MIXELM is an integer flag for the advection solution option. = 3, the hybrid method of characteristics (HMOC)
    # with MOC or MMOC automatically and dynamically selected;
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

    def spd_maker(l_r_c: list, c):
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

    wells_list = [os.path.basename(r.lower()).split('.')[0] for r in obs_files]

    def get_trailing_number(s):
        """Gets last number (str) of a string to get observation ID."""
        m = re.search(r'\d+$', s)
        return int(m.group()) if m else None

    ids = [get_trailing_number(s) for s in wells_list]

    obs_files_sorted = [x for _, x in sorted(zip(ids, obs_files))]

    observations = [mt.load_obs(of) for of in obs_files_sorted]  # Loading observations results
    fields = observations[0].dtype.names
    hey = np.array(
        [
            list(zip(observations[i][fields[1]], observations[i][fields[2]])) for i in range(len(observations))
        ]
    )

    np.save(jp(model_ws, 'bkt'), hey)  # saves raw curves

    return mt
