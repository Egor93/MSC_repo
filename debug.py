import sys
import matplotlib.pyplot as plt
src_path='./src/'
sys.path+=[src_path]
import cf_tree_regression as cft

 # DEFAULT PARAMETERS - global variables
curdir='/home/igor/UNI/Master_Project/Script/Data/'
fname='ncr_pdf_douze_025deg.nc'
goal_var = 'cl_l'
input_vars = ['qsm', 'qtm', 'qlm', 'skew_l', 'var_l', 'var_t', 'tm', 'pm']
add_vars = ['qvlm','qsm']
eval_fraction=0.2
regtype = 'decision_tree'
# ['decision_tree','gradient boost','random forest']
max_depth_in=None

# DATA PREPROCESSING
prepro=cft.DataPrepro(curdir,fname,goal_var,input_vars,add_vars,eval_fraction)
# methods should be in this particular order!!
processed_data = prepro.get_processed_data()

goalvar_pred, goalvar_eval = cft.regression(regtype,processed_data,max_depth_in)

