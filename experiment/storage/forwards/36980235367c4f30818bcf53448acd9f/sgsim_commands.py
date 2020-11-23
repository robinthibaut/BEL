import os

import sgems

nodata = -9966699

os.chdir("C://Users//rthibaut//PycharmProjects//BEL//experiment//storage//forwards//36980235367c4f30818bcf53448acd9f")
sgems.execute('DeleteObjects computation_grid')
sgems.execute('DeleteObjects sgsim')
sgems.execute('DeleteObjects finished')

for file in ['hd.sgems']:
    sgems.execute('LoadObjectFromFile  {}::All'.format(file))

sgems.execute('NewCartesianGrid  computation_grid::150::100::1::10.0::10.0::0::0.0::0.0::0.0')

sgems.execute('RunGeostatAlgorithm  sgsim::/GeostatParamUtils/XML::<parameters>   <algorithm name="sgsim" />    <Grid_Name value="computation_grid" region="" />    <Property_Name value="sgsim" />    <Nb_Realizations value="1" />    <Seed value="492434933" />    <Kriging_Type value="Simple Kriging (SK)" />    <Trend value="0 0 0 0 0 0 0 0 0" />    <Local_Mean_Property value="0" />    <Assign_Hard_Data value="1" />    <Hard_Data grid="hd_grid" property="hd" />    <Max_Conditioning_Data value="15" />    <Search_Ellipsoid value="50 50 50 0 0 0" />    <Use_Target_Histogram value="0" />    <nonParamCdf break_ties="0" filename="0" grid="th" property="thv" ref_on_file="1" ref_on_grid="0">        <LTI_type extreme="-0.1" function="Power" omega="3" />        <UTI_type extreme="0.1" function="Power" omega="0.333" />    </nonParamCdf>    <Variogram nugget="0" structures_count="1">          <structure_1 contribution="1" type="Spherical">            <ranges max="100" medium="50" min="25" />            <angles x="0" y="0" z="0" />        </structure_1>          </Variogram></parameters>')
sgems.execute('SaveGeostatGrid  computation_grid::results.grid::gslib::0::sgsim__real0')