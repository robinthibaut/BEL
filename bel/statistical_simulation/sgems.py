#  Copyright (c) 2020. Robin Thibaut, Ghent University

# This code produces a python code that SGems will understand to run the specified commands.
# Here we produce sgism for the hydraulic conductivity.
# Idea = make Class file containing methods for the simulations and Flopy

import ntpath  # Path for Windows
import os
import subprocess

import numpy as np


# TODO: Separate this script in 2: the first part will be a function, given a python script as argument,
#  will run statistical_simulation The other part will be used to specifically write python code for my simulations.
#  Push all to Github


def path_leaf(path):
    """Extracts file name from a path name"""
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def joinlist(j, mylist):
    """
    Function that joins an array of numbers with j as separator. For example, joinlist('^', [1,2]) returns 1^2
    """
    gp = j.join(map(str, mylist))

    return gp


def so(filename):
    """
    This code transforms the SGems simulations output into a python readable and exploitable format,
    given the statistical_simulation output file as argument
    """

    datac = open(filename, 'r').readlines()
    # Extraction of the grid size from the first lines of the output
    nsimu = int(datac[1])
    # Each simulation result is separated
    head = nsimu + 2
    datac = datac[head:]
    data = [line.split(' ')[:-1] for line in datac if len(line) > 1]
    data = np.array(data, dtype=np.float)
    sims = [data[:, i] for i in range(0, nsimu, 1)]

    return sims


def transform(f, k_mean, k_std):
    """
    Transforms the values of the statistical_simulation simulations into meaningful data
    """

    ff = f * k_std + k_mean

    return 10 ** ff


class SGEMS:

    def __init__(self):
        self.sim_params = []
        self.seed = 0
        # The idea here is to initiate a list that will contain the simulations parameters. I will save it in a text
        # file in dedicated simulations folders for example. I don't know if this 'self' method is optimal yet.

    def gaussian_simulation(self,
                            # core_path='',
                            op_folder='',
                            simulation_name='simulation',
                            output='hydrk',
                            grid=None,
                            fixed_nodes=None,
                            algo='sgsim',
                            number_realizations=1,
                            seed=np.random.randint(10000000),
                            kriging_type='Simple Kriging (SK)',
                            trend=None,
                            local_mean=0,
                            hard_data_grid='',
                            hard_data_property='',
                            assign_hard_data=0,
                            max_conditioning_data=20,
                            search_ellipsoid_geometry=None,
                            target_histogram_flag=0,
                            target_histogram=None,  # min, max, mean, std
                            variogram_nugget=0,
                            variogram_number_structures=1,
                            variogram_structures_contribution=None,
                            variogram_type=None,
                            range_max=None,
                            range_med=None,
                            range_min=None,
                            angle_x=None,
                            angle_y=None,
                            angle_z=None):

        """
        This function produces a python code that SGems will understand to run the specified commands

        :param op_folder:
        :param simulation_name:
        :param output:
        :param grid:
        :param fixed_nodes:
        :param algo:
        :param number_realizations:
        :param seed:
        :param kriging_type:
        :param trend:
        :param local_mean:
        :param hard_data_grid:
        :param hard_data_property:
        :param assign_hard_data:
        :param max_conditioning_data:
        :param search_ellipsoid_geometry:
        :param target_histogram_flag:
        :param target_histogram:
        :param variogram_nugget:
        :param variogram_number_structures:
        :param variogram_structures_contribution:
        :param variogram_type:
        :param range_max:
        :param range_med:
        :param range_min:
        :param angle_x:
        :param angle_y:
        :param angle_z:
        :return:

        """

        if angle_z is None:
            angle_z = [0]
        if angle_y is None:
            angle_y = [0]
        if angle_x is None:
            angle_x = [0]
        if range_min is None:
            range_min = [20]
        if range_med is None:
            range_med = [40]
        if range_max is None:
            range_max = [50]
        if variogram_type is None:
            variogram_type = ['Spherical']
        if variogram_structures_contribution is None:
            variogram_structures_contribution = [1]
        if target_histogram is None:
            target_histogram = [0, 0, 0, 0]
        if search_ellipsoid_geometry is None:
            search_ellipsoid_geometry = [15, 15, 7.5, 0.398941, 0, 0]
        if trend is None:
            trend = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if fixed_nodes is None:
            fixed_nodes = []
        if grid is None:
            grid = [50, 50, 1, 1, 1, 1, 0, 0, 0]

        # FILE NAME
        file_name = os.path.join(op_folder, 'simusgems.py')
        sgf = open(file_name, 'w')

        # Create script file
        run_script = os.path.join(op_folder, 'statistical_simulation.script')
        rscpt = open(run_script, 'w')
        rscpt.write(' '.join(['RunScript', file_name]))
        rscpt.close()

        # Create BAT file
        batch = os.path.join(op_folder, 'RunSgems.bat')
        bat = open(batch, 'w')
        bat.write(' '.join(['cd', op_folder, '\n']))
        bat.write(' '.join(['sgems', 'statistical_simulation.script']))
        bat.close()

        # Number of cells
        ncells = [grid[0] * grid[1] * grid[2], 'NCELLS']
        self.sim_params.append(ncells)

        # Fixed nodes
        fnodes = [fixed_nodes, 'FNODES']
        self.sim_params.append(fnodes)

        # OP FOLDER
        opf = [op_folder, 'OPFOLDER']
        self.sim_params.append(opf)

        # SIMULATION NAME
        value = [simulation_name, 'NAME']  # Name of the simulation grid.
        self.sim_params.append(value)

        # ALGORITHM
        algo = [algo, 'ALGO']  # Which simulation algorithm.
        self.sim_params.append(algo)

        # SIMULATION OUTPUT
        pp = [output, 'OUTPUT']  # Prefix for the simulation output.The suffix __real# is added for each realization.
        self.sim_params.append(pp)

        # HARD DATA GRID
        hd = [hard_data_grid, 'HDG']  # [Hard Data.grid] Name of the grid containing the conditioning data.
        # If no grid is selected, the realizations are unconditional.
        self.sim_params.append(hd)

        # HARD DATA PROPERTY
        hdpp = [hard_data_property, 'HGPP']  # [Hard Data.property] Property for the conditioning data.
        # Only required if a grid has been selected in Hard Data | Object [Hard Data.grid].
        self.sim_params.append(hdpp)

        # ASSIGN HARD DATA
        ahd = [assign_hard_data, 'AHDV']  # Assign hard data to simulation grid [Assign Hard Data] If selected, the
        # hard data are relocated to the simulation grid. The program does not proceed
        # if the assignment fails. This option significantly increases execution speed.
        self.sim_params.append(ahd)

        # MAX CONDITIONING DATA
        mcd = [max_conditioning_data, 'DATACON']  # Maximum number of data to be retained in the search neighborhood.
        self.sim_params.append(mcd)

        # SEARCH ELLIPSOID GEOMETRY
        segp = search_ellipsoid_geometry
        segp = joinlist(' ', segp)
        seg = [segp, 'SEARCHELLIPSOID']  # Parametrization of the search ellipsoid.
        self.sim_params.append(seg)

        # TARGET HISTOGRAM

        minth = target_histogram[0]
        thmin = [minth - 0.1, 'MINTH']
        self.sim_params.append(thmin)
        maxth = target_histogram[1]
        thmax = [maxth + 0.1, 'MAXTH']
        self.sim_params.append(thmax)
        thfile = target_histogram[-1]
        fileth = [thfile, 'THFILE']
        self.sim_params.append(fileth)

        # meanth = target_histogram[2]
        # stdth = target_histogram[3]
        #
        # disth = stats.truncnorm((minth - meanth) / stdth, (maxth - meanth) / stdth, loc=meanth, scale=stdth)
        # valuesth = list(disth.rvs(1000))
        # vth = [valuesth, 'VALUESTH']
        # self.simparams.append(vth)
        # a, b = 500, 600
        # mu, sigma = 550, 30
        # dist = stats.truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
        #
        # values = dist.rvs(1000)

        flag = [target_histogram_flag, 'THF']  # Use Target Histogram [Use Target Histogram] Flag to use the normal
        # score transform. If used, the data are normal score transformed prior to
        # simulation and the simulated field is transformed back to the original space.
        self.sim_params.append(flag)
        th = [target_histogram, 'TARGETHISTO']
        self.sim_params.append(th)

        # If used, the data are normal score transformed prior
        # to simulation and the simulated field is transformed back to the original
        # space. Use Target Histogram [Use Target Histogram] flag to use the normal
        # score transform. The target histogram is parametrized by [nonParamCdf].

        # VARIOGRAM - I implemented a way to insert several structures
        varparams = []
        # Parametrization of the normal score variogram.
        ng = [variogram_nugget, 'NUGGET']  # Value for the nugget effect.
        self.sim_params.append(ng)
        sc = [variogram_number_structures, 'NSTRUCT']  # Number of nested structures, excluding the nugget effect.
        self.sim_params.append(sc)
        ct = [variogram_structures_contribution, 'STRUCTCONT']  # Sill for the current structure.
        self.sim_params.append(ct)
        varparams.append(ct)
        vt = [variogram_type, 'TYPEVG']
        # Type of variogram for the selected structure (spherical, exponential or Gaussian).
        self.sim_params.append(vt)
        varparams.append(vt)
        # Anisotropy
        rmax = [range_max, 'RMAX']
        rmed = [range_med, 'RMED']
        rmin = [range_min, 'RMIN']
        self.sim_params.append(rmax)
        self.sim_params.append(rmed)
        self.sim_params.append(rmin)
        varparams.append(rmax)
        varparams.append(rmed)
        varparams.append(rmin)
        ax = [angle_x, 'ANGLEX']
        ay = [angle_y, 'ANGLEY']
        az = [angle_z, 'ANGLEZ']
        # Maximum, medium and minimum ranges and rotation angles.
        # In 2D, the dip and rake rotations should be 0 and the minimum range must be less than the medium range.
        self.sim_params.append(ax)
        self.sim_params.append(ay)
        self.sim_params.append(az)
        varparams.append(ax)
        varparams.append(ay)
        varparams.append(az)

        def variograms(ns, vparams):

            tpt = '    <structure_S# contribution="STRUCTCONT" type="TYPEVG">            <ranges max="RMAX" medium="RMED" min="RMIN" />            <angles x="ANGLEX" y="ANGLEY" z="ANGLEZ" />        </structure_S#>    '

            vartpt_ = ''

            for i in range(0, ns[0], 1):
                tptu = tpt.replace('S#', str(i + 1))
                for j in range(0, len(vparams), 1):
                    tptu = tptu.replace(vparams[j][1], str(vparams[j][0][i]))
                vartpt_ += tptu

            return vartpt_

        vartpt = [variograms(sc, varparams), 'VARIOGRAMS']

        # self.simparams.append(vartpt)

        # REALIZATIONS
        nr = [number_realizations, 'REALIZATIONS']  # Number of simulations to generate.
        self.sim_params.append(nr)

        # GRID DIMENSIONS
        # [nx, ny, nz, dx, dy, dz, xo, yo, zo]
        grid = [joinlist('::', grid), 'GRID']
        self.sim_params.append(grid)

        # KRIGING
        ktype = [kriging_type, 'KRIGING']  # Select the type of kriging system to be solved
        # at each node along the random path. The simple kriging (SK) mean is set to
        # zero, as befits a stationary standard Gaussian model.
        self.sim_params.append(ktype)
        # SEED
        sv = [seed, 'SEED']  # Seed for the random number generator (preferably a large odd integer).
        self.sim_params.append(sv)
        self.seed = seed
        # TREND
        tv = [trend, 'TREND']  # Trend value
        self.sim_params.append(tv)
        # LOCAL MEAN
        lmpv = [local_mean, 'LMPV']
        self.sim_params.append(lmpv)

        # OUTPUT FILES LIST
        opl = ['::'.join([pp[0] + '__real' + str(i) for i in range(0, nr[0], 1)]), 'OUTLIST']
        # self.simparams.append(opl)

        prms = [[' '.join([str(self.sim_params[i][1]), str(self.sim_params[i][0]), '\n'])
                 for i in range(0, len(self.sim_params), 1)], 'PARAMS']

        self.sim_params.append(prms)

        # Template within which the simulation parameters will get implemented.
        # I implemented a code within this template to create a 'params' file which will contain the information
        # of the simulation.

        template = """\
import sgems as statistical_simulation\n\
import os
os.chdir("OPFOLDER")
#os.chdir('sim')
statistical_simulation.execute('DeleteObjects NAME')\n\
statistical_simulation.execute('DeleteObjects finished')\n\
statistical_simulation.execute('NewCartesianGrid  NAME::GRID')\n\
#\n\
nodata = -9966699\n\
statistical_simulation.execute('NewCartesianGrid  hd::GRID')\n\
hard_data = [nodata for i in range(NCELLS)]\n\
fn = FNODES\n\
for n in fn:\n\
    hard_data[n[0]] = n[1]\n\
statistical_simulation.set_property('hd', 'hard', hard_data)\n\
#\n\
#statistical_simulation.execute('NewCartesianGrid  th::GRID')\n\
#thv = VALUESTH\n\
#statistical_simulation.set_property('th', 'thv', thv)\n\
#\n\
statistical_simulation.execute('DeleteObjects finished')\n\
statistical_simulation.execute('RunGeostatAlgorithm  ALGO::/GeostatParamUtils/XML::<parameters>   <algorithm name="ALGO" />    <Grid_Name value="NAME" />    <Property_Name value="OUTPUT" />    <Nb_Realizations value="REALIZATIONS" />    <Seed value="SEED" />    <Kriging_Type value="KRIGING" />    <Trend value="TREND" />    <Local_Mean_Property value="LMPV" />    <Assign_Hard_Data value="AHDV" />    <Hard_Data grid="HDG" property="HGPP" />    <Max_Conditioning_Data value="DATACON" />    <Search_Ellipsoid value="SEARCHELLIPSOID" />    <Use_Target_Histogram value="THF" />    <nonParamCdf ref_on_file="1" ref_on_grid="0" break_ties="0" filename="THFILE" grid="th" property="thv">        <LTI_type function="Power" extreme="MINTH" omega="3" />        <UTI_type function="Power" extreme="MAXTH" omega="0.333" />    </nonParamCdf>    <Variogram nugget="NUGGET" structures_count="NSTRUCT">      VARIOGRAMS      </Variogram></parameters>')\n\
statistical_simulation.execute('SaveGeostatGrid  NAME::OUTPUT.grid::gslib::0::OUTLIST')\n\
#statistical_simulation.execute('SaveGeostatGrid  NAME::OUTPUT.statistical_simulation::s-gems::0::OUTLIST')\n\
mfn = 'OUTPUT' + '.params'\n\
mfile = open(mfn, 'w')\n\
for line in PARAMS:\n\
    mfile.write(line)\n\
mfile.close()\n\
"""

        for i in range(0, len(self.sim_params), 1):  # Replaces the parameters
            template = template.replace(self.sim_params[i][1], str(self.sim_params[i][0]))

        # I want to save the simparams list in the output file so I excluded from it the 'output' and the long
        # 'variogram' arguments.
        template = template.replace(vartpt[1], vartpt[0])
        template = template.replace(opl[1], opl[0])

        # print(template)

        sgf.write(template)
        sgf.close()

        # shutil.copyfile(file_name, jp(op_folder, path_leaf(file_name)))

        subprocess.call([batch])  # Opens the BAT file

    def SNESIM(self,
               tifile='ti1channel - Copy.grid',
               core_path='',
               op_folder='',
               simulation_name='SNESIM',
               output='hk',
               grid=[250, 250, 1, 5, 5, 5, 0, 0, 0],
               algo='snesim_std',
               number_realizations=1,
               cmin=1,
               cma=0,
               recri=-1,
               reitn=1,
               nmultigrid=3,
               debug=0,
               subgridchoice=0,
               expandiso=1,
               expandaniso=0,
               anisofactor='',
               affinity=0,
               rotation=0,
               usepresimdata=0,
               useprobfield=0,
               pfppc=0,
               pfppcv='',
               taumodelobject='1 1',
               useverticalprop=0,
               gregion='',
               trainingregion='',
               nfacies=2,
               marginalcdf='0.5 0.5',
               trainingproperty='facies',
               seed=np.random.randint(1000000),
               hard_data_grid='',
               hdgregion='',
               hard_data_property='',
               max_conditioning_data=60,
               search_ellipsoid_geometry=[100, 80, 50, 0.5, 0.4, 0.3]):

        simparams = []
        # FILE NAME
        file_name = os.path.join(core_path, 'simusgems.py')
        sgf = open(file_name, 'w')

        # OP FOLDER
        opf = [op_folder, 'OPFOLDER']
        simparams.append(opf)

        # SIMULATION NAME
        value = [simulation_name, 'NAME']  # Name of the simulation grid.
        simparams.append(value)

        # TRAINING IMAGE FILE
        tif = [tifile, 'TIFILE']
        simparams.append(tif)
        # ALGORITHM
        algo = [algo, 'ALGO']  # Which algorithm
        simparams.append(algo)

        # SIMULATION OUTPUT
        pp = [output, 'OUTPUT']  # Prefix for the simulation output.The suffix __real# is added for each realization.
        simparams.append(pp)

        # SEED
        gr = [seed, 'SEED']
        simparams.append(gr)

        # CMIN VALUE
        cm = [cmin, 'CMIN']
        simparams.append(cm)

        # Constraint_Marginal_ADVANCED
        c_m_a = [cma, 'CMA']
        simparams.append(c_m_a)

        # resimulation_criterion
        rc = [recri, 'RECRI']
        simparams.append(rc)

        # resimulation_iteration_nb
        rin = [reitn, 'REITN']
        simparams.append(rin)

        # Nb_Multigrids_ADVANCED
        nma = [nmultigrid, 'NMULTIGRID']
        simparams.append(nma)

        # Debug_Level
        db = [debug, 'DEBUG']
        simparams.append(db)

        # Subgrid_choice
        sgc = [subgridchoice, 'SUBGRIDCHOICE']
        simparams.append(sgc)

        # expand_isotropic
        eiso = [expandiso, 'EXPANDISO']
        simparams.append(eiso)

        # expand_anisotropic
        eaniso = [expandaniso, 'EXPANDANISO']
        simparams.append(eaniso)

        # aniso_factor value
        af = [anisofactor, 'ANISOFACTOR']
        simparams.append(af)

        # Use_Affinity
        afn = [affinity, 'AFFINITY']
        simparams.append(afn)

        # Use_Rotation
        ur = [rotation, 'ROTATION']
        simparams.append(ur)

        # HARD DATA GRID
        hd = [hard_data_grid, 'HDG']  # [Hard Data.grid] Name of the grid containing the conditioning data.
        # If no grid is selected, the realizations are unconditional.
        simparams.append(hd)

        # Hard_Data grid region
        hdgr = [hdgregion, 'HDREGION']
        simparams.append(hdgr)

        # HARD DATA PROPERTY
        hdpp = [hard_data_property, 'HDPP']  # [Hard Data.property] Property for the conditioning data.
        # Only required if a grid has been selected in Hard Data | Object [Hard Data.grid].
        simparams.append(hdpp)

        # use_pre_simulated_gridded_data
        upsd = [usepresimdata, 'USEPRESIMDATA']
        simparams.append(upsd)

        # Use_ProbField  value="USEPROBFIELD"
        upf = [useprobfield, 'USEPROBFIELD']
        simparams.append(upf)

        # ProbField_properties count
        pfpc = [pfppc, 'PFPPC']
        simparams.append(pfpc)

        # ProbField_properties count value
        pfpv = [pfppcv, 'PFCV']
        simparams.append(pfpv)

        # TauModelObject
        tmo = [taumodelobject, 'TAUMODELOBJECT']
        simparams.append(tmo)

        # use_vertical_proportion
        uvp = [useverticalprop, 'USEVERTICALPROP']
        simparams.append(uvp)

        # Grid region
        grg = [gregion, 'GREGION']
        simparams.append(grg)

        # Training image region
        trg = [trainingregion, 'TRAINING_REGION']
        simparams.append(trg)

        # Training image property
        trgp = [trainingproperty, 'TRAINING_PP']
        simparams.append(trgp)

        # Nb_Facies
        nf = [nfacies, 'NFACIES']
        simparams.append(nf)

        # Marginal_Cdf
        mcdf = [marginalcdf, 'MARGINALCDF']
        simparams.append(mcdf)

        # MAX CONDITIONING DATA
        mcd = [max_conditioning_data, 'DATACON']  # Maximum number of data to be retained in the search neighborhood.
        simparams.append(mcd)

        # SEARCH ELLIPSOID GEOMETRY
        segp = search_ellipsoid_geometry
        segp = joinlist(' ', segp)
        seg = [segp, 'SEARCHELLIPSOID']  # Parametrization of the search ellipsoid.
        simparams.append(seg)

        # REALIZATIONS
        nr = [number_realizations, 'REALIZATIONS']  # Number of simulations to generate.
        simparams.append(nr)

        # GRID DIMENSIONS
        # [nx, ny, nz, dx, dy, dz, xo, yo, zo]
        grid = [joinlist('::', grid), 'GRID']
        simparams.append(grid)

        # OUTPUT FILES LIST
        opl = ['::'.join([pp[0] + '__real' + str(i) for i in range(0, nr[0], 1)]), 'OUTLIST']
        # self.simparams.append(opl)

        prms = [[' '.join([str(simparams[i][1]), str(simparams[i][0]), '\n']) for i in
                 range(0, len(simparams), 1)], 'PARAMS']
        simparams.append(prms)

        # Template within which the simulation parameters will get implemented.
        # I implemented a code within this template to create a 'params' file which will contain the information
        # of the simulation.
        template = """\
import sgems as statistical_simulation\n\
import os
os.chdir('OPFOLDER')
tin = os.path.join('.', 'ti', 'TIFILE')\n\
tio = open(tin, 'r').readlines()\n\
data = [float(tio[i]) for i in range(3, len(tio), 1)]\n\
os.chdir('sim')\n\
statistical_simulation.execute('DeleteObjects NAME')\n\
statistical_simulation.execute('DeleteObjects finished')\n\
statistical_simulation.execute('NewCartesianGrid  NAME::GRID')\n\
statistical_simulation.execute('NewCartesianGrid  ti::GRID')\n\
statistical_simulation.set_property('ti', 'facies', data)\n\
statistical_simulation.execute('DeleteObjects finished')\n\
statistical_simulation.execute("RunGeostatAlgorithm  snesim_std::/GeostatParamUtils/XML::<parameters>  <algorithm name='snesim_std' />     <Cmin  value='CMIN' />     <Constraint_Marginal_ADVANCED  value='CMA' />     <resimulation_criterion  value='RECRI' />     <resimulation_iteration_nb  value='REITN' />     <Nb_Multigrids_ADVANCED  value='NMULTIGRID' />     <Debug_Level  value='DEBUG' />     <Subgrid_choice  value='SUBGRIDCHOICE'  />     <expand_isotropic  value='EXPANDISO'  />     <expand_anisotropic  value='EXPANDANISO'  />     <aniso_factor  value='ANISOFACTOR' />     <Use_Affinity  value='AFFINITY'  />     <Use_Rotation  value='ROTATION'  />     <Hard_Data  grid='HDG' region='HDREGION' property='HDPP'  />     <use_pre_simulated_gridded_data  value='USEPRESIMDATA'  />     <Use_ProbField  value='USEPROBFIELD'  />     <ProbField_properties count='PFPPC'   value='PFCV'  />     <TauModelObject  value='TAUMODELOBJECT' />     <use_vertical_proportion  value='USEVERTICALPROP'  />     <GridSelector_Sim value='NAME' region='GREGION'  />     <Property_Name_Sim  value='OUTPUT' />     <Nb_Realizations  value='REALIZATIONS' />     <Seed  value='SEED' />     <PropertySelector_Training  grid='ti' region='TRAINING_REGION' property='TRAINING_PP'  />     <Nb_Facies  value='NFACIES' />     <Marginal_Cdf  value='MARGINALCDF' />     <Max_Cond  value='DATACON' />     <Search_Ellipsoid  value='SEARCHELLIPSOID' />  </parameters>")  
statistical_simulation.execute('SaveGeostatGrid  NAME::OUTPUT.out::gslib::0::OUTLIST')\n\
statistical_simulation.execute('SaveGeostatGrid  NAME::OUTPUT.statistical_simulation::s-gems::0::OUTLIST')\n\
mfn = 'OUTPUT' + '.params'\n\
mfile = open(mfn, 'w')\n\
for line in PARAMS:\n\
    mfile.write(line)\n\
mfile.close()\n\
"""

        for i in range(0, len(simparams), 1):  # Replaces the parameters
            template = template.replace(simparams[i][1], str(simparams[i][0]))

        template = template.replace(opl[1], opl[0])
        sgf.write(template)
        sgf.close()
        os.chdir(core_path)
        os.system('RunSgems.bat')  # This opens a bat file I created to run SGems.
