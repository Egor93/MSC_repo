import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree,ensemble
from matplotlib.colors import LogNorm
import time
import json
# my own 'library' below
####################
import sys
# my own 'library' below
#src_path = '../src/'
# Taylor Diagram from Yannick Copin <yannick.copin@laposte.net> 
taylor_path = './Taylor_Diagram' 
# sys.path.append(src_path)
sys.path += [taylor_path]
####################
import cf_tree_regression as cft
from importlib import reload
from taylorDiagram import TaylorDiagram as taylor

#curdir = '/home/igor/UNI/Master_Project/001_Code/002_Data/'
#curdir = '/home/egordeev/002_Data'
curdir = cft.get_config_params()
#subdomain_sizes = ['1','05','025','0125']
subdomain_sizes = ['1','05']
fnames = [f'ncr_pdf_douze_{i}deg.nc' for i in subdomain_sizes ]
goal_var = 'cl_l'
input_vars = ['qsm', 'qtm', 'qlm', 'skew_l', 'var_l', 'var_t', 'tm', 'pm']
add_vars = [['qvlm','qsm'],[]]
eval_fraction=0.2
regtypes = ['decision_tree','gradient_boost','random_forest']
# max_depth_in=None
max_depth_in=10

# dictionary of ML perfomance samples, will be later used for Taylor diagram
samples = dict()
references = dict()
for size in subdomain_sizes:
    fname = f'ncr_pdf_douze_{size}deg.nc' 
    sampvals = []
    refvals = []
    print(f'-------PROCESS_FILE{fname}--------') 

    for regtype in regtypes:
        for add_var in add_vars:
            
            print(f'-------REGTYPE={regtype}/////ADDVAR={add_var}--------') 

            # DATA PREPROCESSING
            prepro=cft.DataPrepro(curdir,fname,goal_var,input_vars,add_var,eval_fraction)
            # methods should be in this particular order!!
            processed_data = prepro.get_processed_data(split_randomly=True)
            
            # REGRESSION
            goalvar_pred, goalvar_eval = cft.regression(regtype,processed_data,max_depth_in)

            # calculate performance estimators - std and correlation coeff
            # goalvar_pred -  ndarray of predicted(using regression) goal variable
            corrmat=np.corrcoef(goalvar_pred, goalvar_eval)
            corr=corrmat[0,1]
            std = np.std(goalvar_pred)
            # goalvar_eval  -  Masked array of goal variable read from netcdf douze file
            refstd = np.std(goalvar_eval)
            if add_var: # if there are some additional variables
                sampvals.append([std, corr, f'{regtype}_satur_deficit'])
                refvals.append([np.float64(refstd), f'{regtype}_satur_deficit'])
            else: # no additional variables - no saturation deficit
                sampvals.append([std, corr, f'{regtype}'])
                refvals.append([np.float64(refstd), f'{regtype}'])

    samples[size] = sampvals
    references[size] = refvals 

package = dict()
package['samples'] = samples
package['references'] = references
# save to JSON file for further visualization
with open('ML_performance_out.json','w') as jsonfile:
    json.dump(package,jsonfile)



