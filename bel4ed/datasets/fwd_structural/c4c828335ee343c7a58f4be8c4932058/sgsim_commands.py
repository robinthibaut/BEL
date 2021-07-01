import os

import sgems

nodata = -9966699

os.chdir("C://Users//Robin//PycharmProjects//BEL4ED//bel4ed//datasets//fwd_structural//c4c828335ee343c7a58f4be8c4932058")
sgems.execute("DeleteObjects computation_grid")
sgems.execute("DeleteObjects sgsim")
sgems.execute("DeleteObjects finished")

for file in ['hd.sgems']:
    sgems.execute("LoadObjectFromFile  {}::All".format(file))

sgems.execute("NewCartesianGrid  computation_grid::150::100::1::10.0::10.0::0::0.0::0.0::0.0")

sgems.execute('RunGeostatAlgorithm  sgsim::/GeostatParamUtils/XML::<parameters>   <algorithm name="sgsim" />    <Grid_Name region="" value="computation_grid" />    <Property_Name value="sgsim" />    <Nb_Realizations value="1" />    <Seed value="947791475" />    <Kriging_Type value="Simple Kriging (SK)" />    <Trend value="0 0 0 0 0 0 0 0 0" />    <Local_Mean_Property value="0" />    <Assign_Hard_Data value="1" />    <Hard_Data grid="hd_grid" property="hd" />    <Max_Conditioning_Data value="15" />    <Search_Ellipsoid value="50 50 50 0 0 0" />    <Use_Target_Histogram value="0" />    <nonParamCdf break_ties="0" filename="0" grid="th" property="thv" ref_on_file="1" ref_on_grid="0">        <LTI_type extreme="-0.1" function="Power" omega="3" />        <UTI_type extreme="0.1" function="Power" omega="0.333" />    </nonParamCdf>    <Variogram nugget="0" structures_count="1">          <structure_1 contribution="1" type="Spherical">            <ranges max="325.61212903005145" medium="50" min="25" />            <angles x="-13.158454111333725" y="0" z="0" />        </structure_1>          </Variogram></parameters>')
sgems.execute('SaveGeostatGrid  computation_grid::results.grid::gslib::0::sgsim__real0')
if "kriging" in "sgsim":  # Save variance grid
    sgems.execute('SaveGeostatGrid  computation_grid::results_var.grid::gslib::1::sgsim__real0_krig_var')