import pandas as pd
import itertools
import argparse

def get_arg_params():
    """
    process argument parameters provided by the call of this .py script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--output_file",required = True, help = "output setup.csv file")
    parser.add_argument("-r","--rootvars",required = True, help = "rootvariables for permutation")
    parser.add_argument("-e","--extravars",required = True, help = "extra variables for permutation")
    parser.add_argument("-s","--subdomain_sizes",nargs = '+',required = True, help = "list of subdomain sizes in string format")
    parser.add_argument("-t","--regtypes",required = True, help = "list of regression types")

    # should be called like ML_performance.py -s 1 05 025 0125
    args = parser.parse_args()
    output_file = args.output_file
    rootvars = tuple(str.split(args.rootvars,sep=','))
    extravars = tuple(str.split(args.extravars,sep=','))
    subdomain_sizes = (tuple(args.subdomain_sizes),)
    regtypes = (tuple(str.split(args.regtypes,sep=',')),)

    return  output_file ,rootvars, extravars,subdomain_sizes, regtypes


def uniqify(perm):
    '''
    To get unique permutations from 
    the list of all possible permutations
    '''
    unique_perm = []
    for p in perm:
        if not set(p) in unique_perm:
            unique_perm.append(set(p))
    # convert to list of tuples from list of sets
    unique_tuples = [tuple(i) for i in unique_perm]

    return unique_tuples


# get all the permutations
def input_combinations(root_inputvars, extra_inputvars):
    # add extra variable repeatedly, moving from 4 root vars to 8 vars
     
    sequence_pack = dict()
    nextra_vars = len(extra_inputvars)

    # add 1 additional variable on the first round, 2 vars on the 2nd and so on
    for n_vars in range(1,nextra_vars+1):
        permutations = list(itertools.permutations(extra_inputvars,n_vars))
        # need only distinct combinations of extra variables to reduce n of simulations
        permutations_unique = uniqify(permutations)    
    
        for permutation in permutations_unique:
            # assign a sequence of integers as a key to each permutation
            # where each integer - index of variable within extra_inputvars 
            index_series = [str(extra_inputvars.index(i)) for i in permutation]
            index_series_str = ''.join(index_series)
            experiment_id = f'R{index_series_str}'
            sequence_pack[experiment_id] = root_inputvars+permutation

    return sequence_pack


def setup_to_DF(inputs_dict, satdeficit, eval_fraction,regtypes, tree_maxdepth,subdomain_sizes):
    '''
    Write down a table of experiments parameters into pandas DataFrame.
    This table will be consequently read by ML_performance.py
    '''
    nexperiments = len(inputs_dict)    
    # check if the default values of correct length were provided
    for var in (satdeficit,eval_fraction,regtypes,tree_maxdepth):
        assert len(var)==nexperiments,"function argument Error, check len of argument!"

    df = pd.DataFrame(columns = ['input_vars_id','input_vars','satdeficit','eval_fraction','regtypes','tree_maxdepth','subdomain_sizes'])
    # inputs_dict - dictionary with permutated input variables combinations of different length
    expkeys = list(inputs_dict.keys())
    
    # for every experiment id number
    for expid in range(nexperiments):
        expkey = expkeys[expid]  
        input_vars = inputs_dict[expkey]
        df.loc[expid] = [expkey,input_vars, satdeficit[expid],eval_fraction[expid],regtypes[expid],tree_maxdepth[expid],subdomain_sizes[expid]]

    return df


def main():
   
    # add_vars = list with additional variables and without
    # saturation deficit - two states - with or without
    #satdeficit = ((True,False),)
    satdeficit = ((False),)
    eval_fraction = (tuple([0.2]),)
    #regtypes = (('decision_tree','gradient_boost','random_forest'),)
    #regtypes = (('decision_tree'),)
    tree_maxdepth = (tuple([10]),)

    # Split variables into root part(defualt sequence) and extra_variables to add
    # output_csv = 'data/input/setup.csv'
    output_csv,root_inputvars,extra_inputvars,subdomain_sizes,regtypes  = get_arg_params()
    print(root_inputvars,extra_inputvars)
    #root_inputvars = tuple(['qtm','qsm','pm','tm'])
    #extra_inputvars = tuple(['qlm', 'skew_l', 'var_l', 'var_t'])

    # generate all possible sequences of adding extra_inputvars to the root variables
    # put these sequences into dictionary, where keys are experiments id's(represent variables sequence)
    inputs_dict = input_combinations(root_inputvars, extra_inputvars)
    # nexp - number of experiments to run
    nexp = len(inputs_dict)

    # multiply arguments by the number of experiments if necessary
    df = setup_to_DF(inputs_dict, satdeficit*nexp,eval_fraction*nexp,regtypes*nexp,tree_maxdepth*nexp,subdomain_sizes*nexp)
    # write experiment setup table down to the csv file which will be read by ML_performance.py 
    df.to_csv(output_csv,sep='\t')


# if .py script called directly instead of being imported
if __name__=="__main__":

    main()
