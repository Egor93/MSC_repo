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
    parser.add_argument("-e","--extravars",nargs='?',required = False, help = "extra variables for permutation")
    parser.add_argument("-s","--subdomain_sizes",nargs = '+',required = True, help = "list of subdomain sizes in string format")
    parser.add_argument("-t","--regtypes",required = True, help = "list of regression types")
    parser.add_argument("-N","--nexprepeat",required = True,type=int, help = "how many times should each experiment be repeated")

    # should be called like ML_performance.py -s 1 05 025 0125
    args = parser.parse_args()
    output_file = args.output_file
    nexprepeat = args.nexprepeat
    rootvars = tuple(str.split(args.rootvars,sep=','))

    if args.extravars == "None" or args.extravars == "none" or args.extravars == None:
        extravars = None
    else:
        parsed_ev = tuple(str.split(args.extravars,sep=','))
        # if parsed list is empty , e.g. "" or " " was provided
        if parsed_ev:
            extravars = parsed_ev
        else:
            extravars = None

    subdomain_sizes = (tuple(args.subdomain_sizes),)
    regtypes = (tuple(str.split(args.regtypes,sep=',')),)

    return  output_file ,rootvars, extravars,subdomain_sizes, regtypes,nexprepeat


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
    # if extra inputvariables were provided
    if extra_inputvars:
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
    else:
        experiment_id = f'R'
        sequence_pack[experiment_id] = root_inputvars

    return sequence_pack


def setup_to_DF(expkeys,invars_list, satdeficit, eval_fraction,regtypes, tree_maxdepth,subdomain_sizes):
    '''
    Write down a table of experiments parameters into pandas DataFrame.
    This table will be consequently read by ML_performance.py
    '''
    nexperiments = len(expkeys)    
    # check if the default values of correct length were provided
    for var in (satdeficit,eval_fraction,regtypes,tree_maxdepth):
        assert len(var)==nexperiments,"function argument Error, check len of argument!"

    df = pd.DataFrame(columns = ['input_vars_id','input_vars','satdeficit','eval_fraction','regtypes','tree_maxdepth','subdomain_sizes'])
    # inputs_dict - dictionary with permutated input variables combinations of different length
    
    # for every experiment id number
    for expid in range(nexperiments):
        df.loc[expid] = [expkeys[expid],invars_list[expid], satdeficit[expid],eval_fraction[expid],regtypes[expid],tree_maxdepth[expid],subdomain_sizes[expid]]

    return df


def main():
   

    # Split variables into root part(defualt sequence) and extra_variables to add
    # output_csv = 'data/input/setup.csv'
    output_csv,root_inputvars,extra_inputvars,subdomain_sizes,regtypes,nexprepeat  = get_arg_params()
    print(f'INPUT VARIABLES:{root_inputvars},EXTRA VARIABLES:{extra_inputvars}')
    #root_inputvars = tuple(['qtm','qsm','pm','tm'])
    #extra_inputvars = tuple(['qlm', 'skew_l', 'var_l', 'var_t'])

    # generate all possible sequences of adding extra_inputvars to the root variables
    # put these sequences into dictionary, where keys are experiments id's(represent variables sequence)
    inputs_dict = input_combinations(root_inputvars, extra_inputvars)
    # nexp - number of experiments to run
    nexp = len(inputs_dict)

    if nexprepeat==0:
        # no repetitions of experiments allowed
        expkeys = list(inputs_dict.keys())
        invars_list = list(inputs_dict.values())
        # saturation deficit - two states - with or withot ((True,False),)
        satdeficit = ((False),)
        eval_fraction = (tuple([0.2]),)
        tree_maxdepth = (tuple([10]),)
    elif nexprepeat>0:
        print(f'repeat the same setup N={nexprepeat} times')
        nreps = nexprepeat # repeat the same experiment n times
        expkeys = list(inputs_dict.keys())*nreps
        nexpkeys = len(expkeys)
        invars_list = list(inputs_dict.values())*nexpkeys
        subdomain_sizes *= nexpkeys
        regtypes *= nexpkeys
        satdeficit = ((False),)*nexpkeys
        eval_fraction = (tuple([0.2]),)*nexpkeys
        tree_maxdepth = (tuple([10]),)*nexpkeys

    # multiply arguments by the number of experiments if necessary
    df = setup_to_DF(expkeys,invars_list, satdeficit,eval_fraction,regtypes,tree_maxdepth,subdomain_sizes)
    # write experiment setup table down to the csv file which will be read by ML_performance.py 
    df.to_csv(output_csv,sep='\t')


# if .py script called directly instead of being imported
if __name__=="__main__":

    main()
