import matplotlib.pyplot as plt
import numpy as np
import itertools
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


def string_to_touple(inputstr,dtype=None):
    if type(inputstr)!= str:
        inputstr=str(inputstr)
    cleanstr = inputstr.replace('\'','').replace(' ','').strip('()[],')
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

        setup_params['input_vars_id']=string_to_touple(exp_package.input_vars_id)
        setup_params['input_vars']=string_to_touple(exp_package.input_vars)
        setup_params['satdeficit']=string_to_touple(exp_package.satdeficit,dtype=bool)
        setup_params['regtypes']=string_to_touple(exp_package.regtypes)
        setup_params['eval_fraction']=string_to_touple(exp_package.eval_fraction,dtype=float)
        setup_params['tree_maxdepth']=string_to_touple(exp_package.tree_maxdepth,dtype=int)
        setup_params['subdomain_sizes']=string_to_touple(exp_package.subdomain_sizes,dtype=str)
   
        yield setup_params


def samesetup_in_series(existing_series,singlexp_setup):
    '''
    DESCRIPTION
        Check if existing_series from particular existing expout_{input_vars_id}.csv 
        already exist, in such case skip the experiment, proceed to the next one.
        If not run the experiment and append the result to the df from existing csv file.
    OUTPUT
        returns True if the setups are the same
                False - if at least one parameter differes
    '''
    # if same combination of input vars
    if existing_series['satdeficit']!=singlexp_setup['satdeficit']: 
        return False
    elif existing_series['eval_fraction']!=singlexp_setup['eval_fraction']:
        return False
    elif existing_series['regtypes']!=singlexp_setup['regtype']:
        return False
    elif existing_series['tree_maxdepth']!=singlexp_setup['tree_maxdepth']:
        return False
    elif existing_series['subdomain_sizes']!=float(singlexp_setup['size']):
        return False
    else:
        # in case of all setup params are the same
        return True

   
def samesetup_in_df(existing_df,singlexp_setup):
    '''
    DESCRIPTION
        If the current experiment setup already exists as some particular row withing 
        existing csv output file. In such case we need to skip the experiment
        df - dataframe, consists of rows (series)
    '''
    for index,series in existing_df.iterrows():
        # if setup parameters within particular row and current experiment settings are the same
        # meaning that such experiment was already executed and results saved in corresponding CSV
        if samesetup_in_series(series,singlexp_setup):
            return True

    return False


def pack_result(expset_setup,singlexp_setup,expresults,result_cols):
    '''
    DESCRIPTION
        pack the experiment results,prepare for  the pandas dataframe
    INPUT
        expset_setup - setup parameters of the set of experiments
        singlexp_setup - setup parameters of the single experiment,a member of a set
        expresults - results of the single experiment run
    '''
    # write output

    expresults_packed = [
            expset_setup['input_vars_id'][0],
            [",".join(expset_setup['input_vars'])],
            singlexp_setup['satdeficit'],
            singlexp_setup['eval_fraction'],
            singlexp_setup['regtype'],
            singlexp_setup['tree_maxdepth'],
            str(singlexp_setup['size']),
            expresults['refstd'],
            expresults['samplestd'],
            expresults['samplecorr'],
            str(expresults['regression_time']),
            ]
    expdict = dict(zip(result_cols,expresults_packed))
    expresult_series = pd.Series(data=expdict)

    return expresult_series 
            

def run_experiment(netcdfpath, singlexp_setup,split_randomly=True):

    # CREATE EXPERIMENT OBJECT
    experiment=nctree.SingleExperiment(netcdfpath, singlexp_setup)
    # PREPARE EXPERIMENT DATA
    processed_data = experiment.process_input(split_randomly)
    # RUN EXPERIMENT -  REGRESSION
    goalvar_pred, goalvar_eval,regression_time = experiment.regression(processed_data)
    # ESTIMATE SKILL OF REGRESSION 
    samplecorr, samplestd, refstd = experiment.estimate_skill(goalvar_pred, goalvar_eval)
    
    expresults = {'samplecorr':samplecorr, 'samplestd':samplestd, 'refstd':refstd, 'regression_time':regression_time}

    return expresults


def run_set(df_existing,expset_setup,netcdfdir,csvout_dir,csvout_name):
    '''
    DESCRIPTION
        generates CSV output file for the set of experiments
        CSV file contains setup params as well as regression skill estimation, e.g. STD, VAR
        set of experiments - one row within setup.csv, superset setup file
    INPUT
        df_existing - None or pandas Dataframe from existing CSV output of current set of experiments
        expset_setup - parameters setup for the current set of experiments 
        netcdfdir - directory with input netCDF files to train ML algorithm on
        csvout_dir- directory to store results of the set of experiments
    '''
    result_cols = ['input_vars_id','input_vars','satdeficit','eval_fraction','regtypes','tree_maxdepth','subdomain_sizes',
                            'refstd','samplestd','samplevar','exectime']
    df_setresult = pd.DataFrame(columns = result_cols)

    expindex = 0 # to used for appending rows to the df.loc[expindex]
    for size in expset_setup['subdomain_sizes']:
        netcdfname = f'ncr_pdf_douze_{size}deg.nc' 
        netcdfpath = os.path.join(netcdfdir,netcdfname)

        for regtype in expset_setup['regtypes']:
            for eval_fraction in expset_setup['eval_fraction']:
                for tree_maxdepth in expset_setup['tree_maxdepth']:
                    for satdeficit in expset_setup['satdeficit']:
                        
                        input_vars = expset_setup['input_vars']
                        input_vars_id = expset_setup['input_vars_id'][0]
                        # parameters of a particular experiment,member of a set of experiments
                        singlexp_setup = {'input_vars':input_vars,'satdeficit':satdeficit,
                                'eval_fraction':eval_fraction,'regtype':regtype,'tree_maxdepth':tree_maxdepth,'size':size} 
                        
                        ######### UPDATE CSV BRANCHING ##########
                        # if there exists output for the same set of experiments (with the same ID) 
                        if df_existing is not None:
                            # if current experiment setup NON-UNIQUE,same as already existing in CSV
                            if samesetup_in_df(df_existing,singlexp_setup):
                                # proceed to the next,unique experiment setup
                                print(f"skipping experiment N {expindex},results are already in {csvout_name}")
                                continue
                            else:
                                # but the current experiment has UNIQUE setup params
                                expresults = run_experiment(netcdfpath,singlexp_setup,split_randomly=True)
                                expresults_series = pack_result(expset_setup,singlexp_setup,expresults,result_cols)
                                df_setresult = df_setresult.append(expresults_series, ignore_index=True)
                                # concatenate new experiments to the existing set results
                                df_updated = pd.concat([df_existing,df_setresult])
                        else:
                            # if there are no previous output CSV file for the current set of experiments
                                expresults = run_experiment(netcdfpath,singlexp_setup,split_randomly=True)
                                expresults_series = pack_result(expset_setup,singlexp_setup,expresults,result_cols)
                                df_setresult = df_setresult.append(expresults_series, ignore_index=True)
                                df_updated = df_setresult

                        #If the CSV file already exists, it will be overwritten. 
                        #This is done to preserve at least some results in case of a crash of hours-long simulation
                        csvout_path = os.path.join(csvout_dir,csvout_name)
                        df_updated.to_csv(csvout_path,sep='\t')

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
        expset_setup = next(setup_generator) 

        # if expriments for current experiment set, e.g. R0 were already generated
        df_existing= None # assume by default that it wasn't
        input_vars_id = expset_setup['input_vars_id'][0]

        # set of experiments is uniquely indexed by input_vars_id - combination of input variables
        # every row of df_input is a set of experiments, written to separate CSV
        csvout_name = f'expset_{input_vars_id}.csv'

        # take care of all possible permutations of digits in the id part of the CSV filename 
        # e.g. R01==R10
        id_digits=[i for i in input_vars_id.strip('R')]
        permutated_names = set(f'expset_R{"".join(i)}.csv' for i in itertools.permutations(id_digits))
        present_csvnames = set(os.listdir(csvout_dir))
        intersection_name = present_csvnames.intersection(permutated_names)
        assert len(intersection_name) < 2, f'>1 CSV ouput files {intersection_name} for the same ID!; make 1 CSV per ID!'

        # if csvout_name exists for current variable ID
        if intersection_name:
            print(f'intersection of possible and existing CSVout names is {intersection_name}')
            csvout_name = intersection_name.pop()
            csvout_path = os.path.join(csvout_dir,csvout_name)
            # index_col = 0 because index can be non-unique, e.g. 0,1,2,0
            df_existing = pd.read_csv(csvout_path,index_col=0,sep='\t')
            
        # check if the file with same input_vars_id already has been generated
        # if yes throw away set params from expset_setup which already have been executed
        run_set(df_existing,expset_setup,netcdfdir,csvout_dir,csvout_name)
        #fnames = [f'ncr_pdf_douze_{i}deg.nc' for i in expset_setup['subdomain_sizes'] ]


def main():

    ###### SETUP FROM CALLING FUNCTION ########
    netcdfdir, setup_csv, csvout_dir= get_arg_params() 

    ######          LOCAL SETUP        #######
    experiment_params = ['input_vars_id','input_vars','satdeficit','eval_fraction','regtypes','tree_maxdepth','subdomain_sizes']

    run_superset(experiment_params,netcdfdir,setup_csv,csvout_dir)


# if .py script called directly instead of being imported
if __name__=="__main__":

    main()
