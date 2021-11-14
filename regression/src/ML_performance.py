import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import os
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
    parser.add_argument("-n","--netcdfdir",required = True, help = "folder containing input NetCDF files")
    parser.add_argument("-s","--setup_csv",required = True, help = "setup csv file, superset of experiments")
    parser.add_argument("-o","--csvout_dir",required = True, help = "directory to store output csv file to")

    # should be called like ML_performance.py -s 1 05 025 0125
    args = parser.parse_args()
#    subdomain_sizes = args.subdomain_sizes
    netcdfdir = args.netcdfdir
    setup_csv = args.setup_csv
    csvout_dir = args.csvout_dir

    return netcdfdir, setup_csv, csvout_dir 



#netcdfdir= '/home/igor/UNI/Master_Project/001_Code/002_Data/'
# netcdfdir= '/home/egordeev/002_Data'

#TODO: read all the stuff below from data/input/setup/setup.csv !! Including subdomain sizes

def string_to_touple(inputstr,dtype=None):
    if type(inputstr)!= str:
        inputstr=str(inputstr)
    cleanstr = inputstr.replace('\'','').replace(' ','').strip('(),')
    tupleout = cleanstr.split(',')
    
    if dtype:
        if dtype==bool:
            # eval() is better then using bool(" False") which will return True
            tupleout = [eval(i) for i in tupleout]
        else:
            tupleout = [dtype(i) for i in tupleout]

    return tupleout


def setup_sequence(df_input,nsets):
    '''
    Yields set of experiment parameters 
    Later these parameters will be used to iterate over
    and run ML experiments
    '''

    # extract the experiment parameters for each run (each DF row)
    for exp in range(nsets):
        setup_params = dict()
        exp_package = df_input.iloc[exp]

        setup_params['input_vars']=string_to_touple(exp_package.input_vars)
        setup_params['input_vars_id']=string_to_touple(exp_package.input_vars_id)
        setup_params['satdeficit']=string_to_touple(exp_package.satdeficit,dtype=bool)
        setup_params['regtypes']=string_to_touple(exp_package.regtypes)
        setup_params['eval_fraction']=string_to_touple(exp_package.eval_fraction,dtype=float)
        setup_params['tree_maxdepth']=string_to_touple(exp_package.tree_maxdepth,dtype=int)
        setup_params['subdomain_sizes']=string_to_touple(exp_package.subdomain_sizes,dtype=str)
   
        yield setup_params


def run_set(currsetup,netcdfdir,csvout_dir,goal_var='cl_l'):
    '''
    DESCRIPTION
        generates CSV output file for the set of experiments
        CSV file contains setup params as well as regression skill estimation, e.g. STD, VAR
        set of experiments - one row within setup.csv, superset setup file
    INPUT
        currsetup - parameters setup for the current set of experiments 
        netcdfdir - directory with input netCDF files to train ML algorithm on
        csvout_dir- directory to store results of the set of experiments
    '''
    df_setresult = pd.DataFrame(columns = ['input_vars_id','input_vars','satdeficit','eval_fraction','regtypes','tree_maxdepth','subdomain_sizes',
                                        'refstd','samplestd','samplevar','exectime'])

    for size in currsetup['subdomain_sizes']:
        netcdfname = f'ncr_pdf_douze_{size}deg.nc' 
        netcdfpath = os.path.join(netcdfdir,netcdfname)

        sampvals = []
        refvals = []
        expindex = 0 # to used for appending rows to the df.loc[expindex]

        for regtype in currsetup['regtypes']:
            for eval_fraction in currsetup['eval_fraction']:
                for tree_maxdepth in currsetup['tree_maxdepth']:
                    for satdeficit in currsetup['satdeficit']:
                        if satdeficit:
                            add_vars = ['qvlm','qsm']
                        else:
                            add_vars = []
                        
                        input_vars = currsetup['input_vars']
                        vars_dict = {"input_vars":input_vars,"add_var":add_vars,"goal_var":goal_var}

                        # CREATE EXPERIMENT OBJECT
                        experiment=nctree.SingleExperiment(netcdfpath, vars_dict, eval_fraction, regtype, tree_maxdepth,resolution = size)
                        # PREPARE EXPERIMENT DATA
                        processed_data = experiment.process_input(split_randomly=True)
                        # RUN EXPERIMENT -  REGRESSION
                        goalvar_pred, goalvar_eval,regression_time = experiment.regression(processed_data)
                        # ESTIMATE SKILL OF REGRESSION 
                        samplecorr, samplestd, refstd = experiment.estimate_skill(goalvar_pred, goalvar_eval)

                        # write output
                        input_vars_id = currsetup['input_vars_id'][0]
                        expresult = [
                                input_vars_id,
                                [",".join(input_vars)],
                                satdeficit,
                                eval_fraction,
                                regtype,
                                tree_maxdepth,
                                str(size),
                                refstd,
                                samplestd,
                                samplecorr,
                                str(regression_time),
                                ]

                        df_setresult.loc[expindex] = expresult 
                        
                        # every row of df_input is a set of experiments, written to separate CSV
                        # set of experiments is uniquely indexed by input_vars_id - combination of input variables
                        csvout_name = f'expset_{input_vars_id}.csv'
                        csvout_path = os.path.join(csvout_dir,csvout_name)
                        #If the CSV file already exists, it will be overwritten. 
                        #This is done to preserve at least some results in case of a crash of hours-long simulation
                        df_setresult.to_csv(csvout_path,sep='\t')
                        # incremental increase of experiment index to be used with df.loc[]
                        expindex += 1


def run_superset(experiment_params,netcdfdir,setup_csv,csvout_dir):
    '''
    DESCRIPTION
        Superset of experiments - all experiment sets withing setup.csv file
        One superset - one setup.csv file. Each row of setup.csv - set of experiments parameters
    INPUT
        experiment_params - list of selected colnames for setup.csv superset parameter file
        netcdfidr         - location of input NetCDFs to train ML on
        setup_csv         - path to the setup.csv file, superset parameter file
        csvout_dir        - directory to store results of the set of experiments
    '''
    df_input = pd.read_csv(setup_csv,sep='\t',usecols = experiment_params)
    # nsets - number of sets (rows) of parameters within current setup.csv superset file
    nsets = df_input.shape[0]
    setup_generator = setup_sequence(df_input,nsets) 

    for iset in range(nsets):
        # yield current setup dictionary, containing all experiment set parameters
        currsetup = next(setup_generator) 
        run_set(currsetup,netcdfdir,csvout_dir,goal_var='cl_l')
        #fnames = [f'ncr_pdf_douze_{i}deg.nc' for i in currsetup['subdomain_sizes'] ]


def main():

    ###### SETUP FROM CALLING FUNCTION ########
    netcdfdir, setup_csv, csvout_dir= get_arg_params() 

    ######          LOCAL SETUP        #######
    experiment_params = ['input_vars','input_vars_id','satdeficit','eval_fraction','regtypes','tree_maxdepth','subdomain_sizes']

    run_superset(experiment_params,netcdfdir,setup_csv,csvout_dir)


# if .py script called directly instead of being imported
if __name__=="__main__":

    main()
