#  Copyright (c) 2021. Robin Thibaut, Ghent University

from os.path import join as jp

import flopy
import numpy as np


def backtrack(flowmodel, exe_name: str, load: bool = False):
    """
    Function to implement Modpath 7 backtracking simulation.
    :param  flowmodel: Modflow model
    :param exe_name: MP7 executable file path
    :param load: bool, option to load the model instead of running it
    :return: if load is False, returns the particles endpoints
             if load is True, returns the model itself
    """
    model_name = flowmodel.name
    # MODPATH
    mpfname = "_".join([model_name, "mp"])  # Model name
    model_ws = flowmodel.model_ws  # Model working directory
    dis = flowmodel.dis  # DIS package
    wells_data = np.load(jp(model_ws, "spd.npy"))  # Stress period data
    pumping_well_data = wells_data[0]  # Pumping well in first
    pw_lrc = pumping_well_data[0][:3]  # Pumping well layer, row, column
    wn = int(dis.get_node(pw_lrc)[0])  # Pumping well node number

    # Create particle group

    drape = 0
    columncelldivisions = 12
    rowcelldivisions = 12
    layercelldivisions = 1
    # Total number of particles = ccd*rcd*lcd
    # Subdivision data
    sd = flopy.modpath.CellDataType(
        drape=drape,
        columncelldivisions=columncelldivisions,
        rowcelldivisions=rowcelldivisions,
        layercelldivisions=layercelldivisions,
    )
    # Particle data
    p = flopy.modpath.mp7particledata.NodeParticleData(subdivisiondata=sd, nodes=wn)

    pg = flopy.modpath.ParticleGroupNodeTemplate(
        particlegroupname="PG", particledata=p, filename="pg.sloc"
    )

    particlegroups = [pg]

    # Default iface for MODFLOW-2005
    defaultiface = {"RECHARGE": 6, "ET": 6}

    # Create modpath files

    mp = flopy.modpath.Modpath7(
        modelname=mpfname, flowmodel=flowmodel, exe_name=exe_name, model_ws=model_ws
    )
    # Define porosity
    lpf = flowmodel.lpf  # LPF package
    porosity_mp = lpf.sy.array  # Porosity loaded from the LPF package = sy

    flopy.modpath.Modpath7Bas(mp, porosity=porosity_mp, defaultiface=defaultiface)

    simulationtype = "combined"
    trackingdirection = "backward"
    weaksinkoption = "pass_through"
    weaksourceoption = "pass_through"
    budgetoutputoption = "summary"
    budgetcellnumbers = None
    traceparticledata = None
    referencetime = None
    stoptimeoption = "specified"
    stoptime = 30  # Stop time in days
    timepointdata = None
    zonedataoption = "off"
    zones = None

    # Run modpath simulation

    flopy.modpath.Modpath7Sim(
        mp,
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
        zonedataoption=zonedataoption,
        zones=zones,
        particlegroups=particlegroups,
    )

    # Write modpath datasets
    mp.write_input()

    if load:
        return mp
    # Run modpath
    mp.run_model(silent=True)

    # %% Loading MODPATH results

    # Load backward tracking path lines
    # fpth = jp(model_ws, mpfname + '.mppth')
    # gp = flopy.utils.PathlineFile(fpth)
    # pwb = gp.get_destination_pathline_data(dest_cells=wn)

    # Load backward tracking endpoints
    modpath_files = jp(model_ws, ".".join([mpfname, "mpend"]))
    e = flopy.utils.EndpointFile(modpath_files)
    # noinspection PyTypeChecker
    ewb = e.get_destination_endpoint_data(dest_cells=wn, source=True)
    # In this implementation, we are only interested in x y locations.
    xep = ewb.X  # Endpoints x-locations (m)
    yep = ewb.Y  # Endpoints y-locations (m)
    pzone_xy = np.array(list(zip(xep, yep)))

    np.save(jp(model_ws, "tracking_ep"), pzone_xy)

    return pzone_xy
