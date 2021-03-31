# MSC_repo
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-introduction">Project introduction</a>
    </li>
    <li>
      <a href="#used-grids">Used grids</a>
    </li>
    <li>
      <a href="#input-data-description">Input data description</a>
      <ul>
        <li><a href="#files-list">Files list</a></li>
      </ul>
      <ul>
        <li><a href="#variables">Variables</a></li>
      </ul>
    </li>
    <li>
      <a href="#machine-learning">Machine learning</a>
       <ul>
        <li><a href="#tree-regression">Tree regression</a></li>
      </ul>
    </li>
  </ol>
</details>




## Project introduction
Topic: Use machine learning to revisit the data and questions of Griewank, Schemann, Neggers 2018<br>
ICON high resolution (156m) is beneﬁcial to the representation of processes that are subgrid-scale for regular (horizontal resolution of 2.8 km for COSMO) numerical weather forecast and climate models.<br>
PDF cloud scheme parameterizes thermodynamic variables which ICON-LES resolves explicitly. Thus we can compare parametrization with "true" data - 3D high resolution ICON-LES data. The idea of Griewank, Schemann, Neggers 2018 article was to tweak different parameters and degrees of freedom of PDF cloud scheme, also change the grid cell the PDF applied to. These changes produce different results and then compared to "true" distributions of td variables from ICON-LES high resolved data. Also performances of skewed and symmetrical PDFs were compared.
## Used grids
 ICON - LES:<br>
- *horizontal resolution:156-m triangle edge grid cell
- vertical resolution:dz changing from 20m in the near-surface area to the 1000 m at the top. 
- n vertical layers:150, n=150 - surface, n=1 - top of the atmosphere.
- cell shape: triangle<br>
*resolution refers to the square root of the mean cell area in the icosahedral grid of ICON<br>

 CLOUD FRACTION PARAMETRIZATION (Griewank, Schemann, Neggers 2018):<br>
- horizontal resolutions:
  - 110 × 110 km,20 subdomains inside cut out area
  - 55 × 55 km, 80 subdomains inside cut out area
  - 27.5 x 27.5 km, 320 subdomains inside cut out area
  - 13.8 x 13.8 km, 1280 subdomains inside cut out area
  - 7 x 7 km, 5120 subdomains inside cut out area
- vertical resolution:?
- n vertical layers:150, n=150 - surface, n=1 - top of the atmosphere
- cell shape: square longitude latitude subdomains


## Input data description

### Files list
"douze" stands for 12; douze files contain all the values from all 12 snapshots indepenetly<br>
Grid cells called SUBDOMAINS below<br>
![alt text](https://github.com/Egor93/MSC_repo/blob/master/Files_list.png)


ICON-LEM has 156-m triangle edge grid cells<br>
NT=12,12 3-D snapshots simulated by the ICON model in LES mode at 12 point in time<br><br>:
- 24 April 2013  12:35, 18:35;<br>
- 25 April 2013 12:07, 14:07; <br>
- 26 April 2013 18:22, <br>
- 11 May 2013 14:06;<br>
- 29 July 2014 06:00, 14:00; <br>
- 14 August 2014 12:03 18:03 <br>
- 3 June 2016 6:00, 14:00.<br><br>

### Variables
INPUTS FROM LES <br>
*mean below refers to spatial averaging of ICON-LES gridcells within particular "Cloud fraction parametrization" subdomain? <br>
- skew_l : LES liquid water+vapor skewness
- var_l  : Liquid water + vapor variance
- qvlm   : Mean liquid water + vapor mixing ratio [kg/kg]
- qvl_qs : the saturation deficit when qvlm is smaller than qsm ??
- pm     : Mean pressure [Pa]
- tm     : Mean temperature [K]
- zm     : Mean height [m]? Values from 0 to 21000 <br>
q means water content, v is vapor, l is liquid, m is mean. so qvlm is the liquid and gas water content.<br>
- qvlm - qsm is the saturation deficit when qvlm is smaller than qsm 
- qcm    : Mean cloud water mixing ratio ?[kg/kg]
- qsm    : Saturation mixing ratio ?[kg/kg]
- qvm    : Mean water vapour mixing ratio[kg/kg]? 
  

Below are plots of some variable from ncr_pdf_douze_1deg.nc. <br>
There are 20 horizontal subdomains of size 110 × 110 km for each of 12 snapshots. Thus 20*12=240 subdomain columns - Y AXIS. Each of these subdomain columns has nz=150 values of each variable, such as pm. These values were calculated over all the LES gridcells(156m size) in the subdomain.
![alt text](https://github.com/Egor93/MSC_repo/blob/master/pm_tm_1degree.png)
![alt text](https://github.com/Egor93/MSC_repo/blob/master/qvm_1degree.png)

<br><br>
CLOUD FRACTIONS<br>
cl_l   : LES liquid cloud fraction, determined from the number of saturated cells(156m) divided by the total
number of cells per slice(subdomain)<br>
cl_msl : cloud fraction of parametrization V "mean saturation" closure<br>
cl_zsl : cloud fraction of parametrization VI "zero buoyancy" closure<br>
cl_rel : cloud fraction of parametrization II - "relative humidity" closure<br>
cl_zff : cloud fraction of parametrization VIII - no closure<br>
![alt text](https://github.com/Egor93/MSC_repo/blob/master/cl_l_1degree.png)


## Machine learning

### Tree regression
TREE REGRESSION INPUT VARIABLES:<br><br>
'qsm', 'qtm', 'qlm', 'skew_l', 'var_l', 'var_t', 'tm', 'pm' - ICON-LES values averaged within the corresponding "cloud parametrization" subdomain. <br>
TREE REGRESSION GOAL VARIABLE:<br><br>
cl_l   :  liquid cloud fraction of LES, determined from the number of saturated cells(156m) divided by the total
number of cells per slice(subdomain).<br>
Input are variables split into:
- test(evaluation) fraction -  fraction of input/goal variables for evaluation of regression method
- training fraction= 1- test fraction (the rest of the data)
