from os.path import join as jp

import flopy
import numpy as np


def my_backtracking(flowmodel, exe_name):
    model_name = flowmodel.name
    # MODPATH
    mpfname = model_name + '_mp'
    model_ws = flowmodel.model_ws
    dis = flowmodel.dis  # DIS package
    wells_data = np.load(jp(model_ws, 'spd.npy'))
    pumping_well_data = wells_data[0]  # Pumping well in last
    pw_lrc = pumping_well_data[0][:3]
    wn = int(dis.get_node(pw_lrc)[0])

    # Create particle group

    drape = 0
    columncelldivisions = 15
    rowcelldivisions = 15
    layercelldivisions = 1
    # Total number of particles = ccd*rcd*lcd
    # Subdivision data:
    sd = flopy.modpath.CellDataType(drape=drape,
                                    columncelldivisions=columncelldivisions,
                                    rowcelldivisions=rowcelldivisions,
                                    layercelldivisions=layercelldivisions)
    # Particle data
    p = flopy.modpath.mp7particledata.NodeParticleData(subdivisiondata=sd,
                                                       nodes=wn)

    pg = flopy.modpath.ParticleGroupNodeTemplate(particlegroupname='PG',
                                                 particledata=p,
                                                 filename='pg.sloc')

    particlegroups = [pg]

    # Default iface for MODFLOW-2005
    defaultiface = {'RECHARGE': 6, 'ET': 6}

    # Create modpath files

    mp = flopy.modpath.Modpath7(modelname=mpfname,
                                flowmodel=flowmodel,
                                exe_name=exe_name,
                                model_ws=model_ws)

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
    stoptime = 30  # Stop time in days
    timepointdata = None
    zonedataoption = 'off'
    zones = None

    # Run modpath simulation

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
    modpath_files = jp(model_ws, mpfname + '.mpend')
    e = flopy.utils.EndpointFile(modpath_files)
    # noinspection PyTypeChecker
    ewb = e.get_destination_endpoint_data(dest_cells=wn, source=True)
    # In my implementation I'm only interested in x y locations.
    xep = ewb.x  # Endpoints x-locations (m)
    yep = ewb.y  # Endpoints y-locations (m)
    pzone_xy = np.array(list(zip(xep, yep)))

    np.save(jp(model_ws, 'tracking_ep'), pzone_xy)

    return pzone_xy
