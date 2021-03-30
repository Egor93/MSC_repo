# MSC_repo
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-introduction">Project introduction</a>
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
  </ol>
</details>




## Project introduction
Topic: Use machine learning to revisit the data and questions of Griewank, Schemann, Neggers 2018<br>
ICON high resolution (156m) is beneﬁcial to the representation of processes that are subgrid-scale for regular (horizontal resolution of 2.8 km for COSMO) numerical weather forecast and climate models.

## Input data description
### Files list
![alt text](https://github.com/Egor93/MSC_repo/blob/master/Files_list.png)

Grid cells called SUBDOMAINS below<br>
ICON-LEM has 156-m triangle edge grid cells<br>
NT=12,12 3-D snapshots simulated by the ICON model in LES mode<br><br>
24 April 2013  12:35, 18:35;<br>
25 April 2013 12:07, 14:07; <br>
26 April 2013 18:22, <br>
11 May 2013 14:06;<br>
29 July 2014 06:00, 14:00; <br>
14 August 2014 12:03 18:03; <br>
3 June 2016 6:00, 14:00.<br><br>

### Variables
INPUTS FROM LES <br>
skew_l : LES liquid water+vapor skewness<br>
var_l  : Liquid water + vapor variance<br>
qvlm   : Mean liquid water + vapor<br>
pm     : Mean pressure<br>
tm     : Mean temperature<br>
![alt text](https://github.com/Egor93/MSC_repo/blob/master/pm_tm_1degree.png)
![alt text](https://github.com/Egor93/MSC_repo/blob/master/qvm_1degree.png)

<br><br>
CLOUD FRACTIONS<br>
cl_l   : LES liquid cloud fraction<br>
cl_msl : cloud fraction of parametrization V <br>
cl_zsl : cloud fraction of parametrization VI<br>
cl_rel : cloud fraction of parametrization II<br>
cl_zff : cloud fraction of parametrization VIII<br>
![alt text](https://github.com/Egor93/MSC_repo/blob/master/cl_l_1degree.png)




