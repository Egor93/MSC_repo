import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.colors import LogNorm
import time
import json
import argparse
# my own 'library' below
import NetCDFregression.tree as nctree

def get_arg_params():
    """
    process argument parameters provided by the call of this .py script
    """
    parser = argparse.ArgumentParser()
#    parser.add_argument("-s","--subdomain_sizes",nargs = '+',required = True, help = "list of subdomain sizes in string format")
    parser.add_argument("-o","--output_file",required = True, help = "output JSON file")
    parser.add_argument("-i","--netcdfdir",required = True, help = "folder containing input NetCDF files")

    # should be called like ML_performance.py -s 1 05 025 0125
    args = parser.parse_args()
#    subdomain_sizes = args.subdomain_sizes
    output_file = args.output_file
    netcdfdir = args.netcdfdir

    return netcdfdir, output_file 


def skill_estimation(goalvar_pred, goalvar_eval): 
            # calculate performance estimators - std and correlation coeff
            # goalvar_pred -  ndarray of predicted(using regression) goal variable
            corrmat=np.corrcoef(goalvar_pred, goalvar_eval)
            corr=corrmat[0,1]
            std = np.std(goalvar_pred)
            # goalvar_eval  -  Masked array of goal variable read from netcdf douze file
            refstd = np.std(goalvar_eval)

    return corr, std, refstd

#netcdfdir= '/home/igor/UNI/Master_Project/001_Code/002_Data/'
# netcdfdir= '/home/egordeev/002_Data'
netcdfdir,output_file = get_arg_params() 

#TODO: read all the stuff below from data/input/setup/setup.csv !! Including subdomain sizes

def string_to_touple(inputstr,dtype=None):
    cleanstr = inputstr.replace('\'','').strip('(),')
    tupleout = cleanstr.split(',')
    
    if dtype:
        if dtype==bool:
            # eval() is better then using bool(" False") which will return True
            tupleout = [eval(i) for i in tupleout]
        else:
            tupleout = [dtype(i) for i in tupleout]

    return tupleout

# read csv as pandas DF
# create output DF to be written into data/output/setup/setup_out.csv -
experiment_params = ['input_vars','input_vars_id','satdeficit','eval_fraction','regtypes','tree_maxdepth','subdomain_sizes']
df_input = pd.read_csv('data/input/setup/setup.csv',sep='\t',usecols = experiment_params)
nexp = df_input.shape[0] # n of experiments==n rows
for exp in range(nexp):
    exp_package = df_input.iloc[exp]
    # extract the experiment parameters for each run (each DF row)
    input_vars = string_to_touple(exp_package.input_vars)
    satdeficit = string_to_touple(exp_package.satdeficit,dtype=bool)
    regtypes = string_to_touple(exp_package.regtypes)
    eval_fraction = string_to_touple(exp_package.eval_fraction,dtype=float)
    tree_maxdepth = string_to_touple(exp_package.tree_maxdepth,dtype=int)
    

# run the experiment cycle 
# write additional columns to the setup_out.csv

fnames = [f'ncr_pdf_douze_{i}deg.nc' for i in subdomain_sizes ]
goal_var = 'cl_l'
input_vars = ['qsm', 'qtm', 'qlm', 'skew_l', 'var_l', 'var_t', 'tm', 'pm']
# add_vars = list with additional variables and without
add_vars = [['qvlm','qsm'],[]]
eval_fraction=0.2
regtypes = ['decision_tree','gradient_boost','random_forest']

# ML_max_depth=None
ML_max_depth=10

# dictionary of ML perfomance samples, will be later used for Taylor diagram
samples = dict()
references = dict()
for size in subdomain_sizes:
    fname = f'ncr_pdf_douze_{size}deg.nc' 
    abspath = netcdfdir + fname

    sampvals = []
    refvals = []
    #print(f'-------PROCESS_FILE{fname}--------') 

    for regtype in regtypes:
        for add_var in add_vars:
            
            #print(f'-------REGTYPE={regtype}/////ADDVAR={add_var}--------') 

            # DATA PREPROCESSING
            vars_dict = {"input_vars":input_vars,"add_var":add_var,"goal_var":goal_var}
            Regressor=nctree.DataPrepro(abspath, vars_dict, eval_fraction, regtype, ML_max_depth,resolution = size)
            # methods should be in this particular order!!
            processed_data = Regressor.get_processed_data(split_randomly=True)

            # REGRESSION
            goalvar_pred, goalvar_eval = Regressor.regression(processed_data)
            # SKILL ESTIMATION
            corr, std, refstd = skill_estimation(goalvar_pred, goalvar_eval)

            if add_var: # if there are some additional variables
                fname_postfix = 'satur_deficit'
            else:
                fname_postfix = ''

            sampvals.append([std, corr, f'{regtype}{fname_postfix}'])
            refvals.append([np.float64(refstd), f'{regtype}{fname_postfix}'])

    samples[size] = sampvals
    references[size] = refvals 

    package = dict()
    package['samples'] = samples
    package['references'] = references
    # save to JSON file for further visualization
    with open(output_file,'w') as jsonfile:
        json.dump(package,jsonfile)



