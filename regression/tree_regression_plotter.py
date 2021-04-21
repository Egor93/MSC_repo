from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree,ensemble
from matplotlib.colors import LogNorm
import xarray as xr
import time
####################
import sys
# my own 'library' below
src_path='../src/'
sys.path.append(src_path)
####################
import cf_tree_regression as cft

 # DEFAULT PARAMETERS - global variables
datadir='/home/igor/UNI/Master_Project/Script/Data/'
fname='ncr_pdf_douze_025deg.nc'
goal_var = 'cl_l'
input_vars = ['qsm', 'qtm', 'qlm', 'skew_l', 'var_l', 'var_t', 'tm', 'pm']
add_vars = ['qvlm','qsm']
eval_fraction=0.6
regtype = 'decision_tree'
# ['decision_tree','gradient boost','random forest']
max_depth_in=None

# DATA PREPROCESSING
prepro=cft.DataPrepro(datadir,fname,goal_var,input_vars,add_vars,eval_fraction)
# methods should be in this particular order!!
processed_data = prepro.get_processed_data()

# REGRESSION
goalvar_pred, goalvar_eval = cft.regression(regtype,processed_data,max_depth_in)

