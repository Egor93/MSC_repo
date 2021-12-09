import json 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools
import os
import pandas as pd
# Taylor Diagram from Yannick Copin <yannick.copin@laposte.net> 
from external.diagram.taylorDiagram import TaylorDiagram as taylor
import ML_performance as mlp

def get_arg_params():
    # parse input variables
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_dir",required = True, help = "input JSON file name")
    parser.add_argument("-o","--output_dir",required = True, help = "output PNG file")
    # there are two possible output types: singleplot or multiplot
    parser.add_argument("-m","--multiplot",required = True, help = "vals=True/False; output several PNG files, one file per Taylor diagram or single plot")
    parser.add_argument("-N","--nexprepeat",required = True,type=int, help = "how many times should each experiment be repeated")
    parser.add_argument("-R","--root_inputvars",required = True,type=str, help = "root input variables")
    parser.add_argument("-k","--subgroup_key",nargs="?",required = False,type=str, help = "key for binary group comparison")
    parser.add_argument("-v","--subgroup_val",nargs="?",required = False,type=str, help = "value for binary group comparison")

    args = parser.parse_args()
    # input_dir where results of experiments are stored, e.g. as CSV
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    MULTIPLOT = eval(args.multiplot)
    NEXPREPEAT = args.nexprepeat
    ROOT_INPUTVARS = args.root_inputvars
    try:
        SUBGROUP_KEY = args.subgroup_key
        SUBGROUP_VAL = args.subgroup_val
    except NameError:
        # if arguments are not provided, assign default values
        SUBGROUP_KEY = None
        SUBGROUP_VAL = None

    return INPUT_DIR,OUTPUT_DIR,MULTIPLOT,NEXPREPEAT,ROOT_INPUTVARS,SUBGROUP_KEY,SUBGROUP_VAL


def get_rows_cols(nplots):
    '''
    create rectangular plot coordinates for 
    a given number of plots. 
    These coordinates and then used by pyplot
    '''
    root = np.sqrt(nplots)

    if root.is_integer():
        rows = cols = root
    elif np.round(root) == np.ceil(root):
        rows = cols = np.ceil(root)
    else: 
        rows = np.floor(root) # n experiments
        cols = rows + 1

    return int(rows),int(cols)


def get_threadset(rootexpid,sorted_expids):
    '''
    DESCRIPTION
        e.g. get 'R0123', -> 'R012','R013','R023','R123'
    '''
    rootexpid 
    addvars=[i for i in rootexpid.strip('R')]
    nvars = len(addvars)
    perm = list(itertools.permutations(addvars))
    nthreads = len(perm)

    threadset = []
    for indexset in perm:
        thread = []
        for i in range(0,nvars):
            branch = f"R{''.join(sorted(indexset[i:]))}"
            thread.append(branch)
        threadset.append(sorted(thread,key=len,reverse=True))

    #It's possible that not all of the theoretical
    #input values combinations were generated - intersection is necessary

    for i in range(nthreads):
        currthread = threadset[i]
        actual_thread = set(sorted_expids).intersection(set(currthread))
        threadset[i] = sorted(list(actual_thread),key=len,reverse=True)
    
    threadset_corrlen = []  # threadset with correct length
    for tr in threadset:
        if len(tr)==nvars:
            threadset_corrlen.append(tr)
        else:
            print(f"skipping thread {tr}, not full,DBase lacks experiments results")
    

    return threadset_corrlen


def get_rootid(sorted_expids):
    
    # start from the experiment with maximum amount of input vars
    maxlen = np.max([len(i) for i in sorted_expids])
    longest_index = [i for i in sorted_expids if len(i)==maxlen]
    assert len(longest_index)==1 , f"Warning! More then one Longest expindex {longest_index}"
    root_index = longest_index[0]

    return root_index


def get_varslegend(subdf,rootexpid):
    '''
    DESCRIPTION
        get Root and additional variables to show later in the plot legend
    '''
    rootvarsdict = dict()
    addvarsdict = dict()
    rootexp = subdf.loc[subdf['input_vars_id']==rootexpid]
    varslist = mlp.string_to_touple(rootexp['input_vars'].values[0])
    # split the longest list of input vars intp root and additional vars
    naddvars = len(rootexpid.strip('R'))
    rootvarsdict['rootvars'] = varslist[:naddvars]

    for i in range(naddvars):
        try:
            addseries = subdf.loc[subdf['input_vars_id']==f'R{i}']
            expvars = mlp.string_to_touple(addseries['input_vars'].values[0])
            addvarsdict[i] = expvars[-1]
        except:
            print(f'addvar N{i} is not in expresult database, skipping')
            continue

    return rootvarsdict,addvarsdict

def display_text(taylor_diagram,subdf,rootexpid,fixed_params,perplot_key,perplot_val,nexprepeat,title):
    '''
    DESCRIPTION
        plot text boxes and title containing all important information 
        regarding the plot of ML experiment set
        Parameters to show:
        1)perplot parameters - fixed for the image, change from one to another
        2)fixed   parameters - fixed for the set of images
        3)changing parameters- change within particular image
    '''

    ax = taylor_diagram._ax

    ax.set_title(title)
                                
    # rootvarsdict contains root and additional variables
    if nexprepeat==0:
        # if every experiment has unique input values combination
        rootvarsdict,addvarsdict = get_varslegend(subdf,rootexpid)
    else:
        # repeating experiments, same input vars, no additional vars
        rootvarsdict = {'rootvars':subdf.loc[0]['input_vars']}  
        addvarsdict  = dict()
    
    # create text box with explanation
    fixed_str   = '\n'.join((f'{perplot_key}:{perplot_val}',
                                    f"tree maxdepth:{fixed_params['tree_maxdepth']}", 
                                    f"part of input data used for evaluation:{fixed_params['eval_fraction']}",    
                                    f"regression type:{fixed_params['regtypes']}",    
                                    f"saturation deficit input data used:{fixed_params['satdeficit']}"))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # text(x,y - position where to put text 
    ax.text(0.05, 0.95, fixed_str, transform=ax.transAxes, fontsize=14,
            position=(1.0,0.9), bbox=props)

    unfixed_str = '\n'.join((f'root variables',f"{rootvarsdict['rootvars']}",f'additional vars',f'{addvarsdict}'))
    props = dict(boxstyle='round', facecolor='blue', alpha=0.5)
    ax.text(0.05, 0.95, unfixed_str, transform=ax.transAxes, fontsize=14,
            position=(1.0,0.7), bbox=props)




def setup_from_CLI():
    ###### SETUP FROM CALLING FUNCTION ########
    INPUT_DIR,OUTPUT_DIR,MULTIPLOT,NEXPREPEAT,ROOT_INPUTVARS,SUBGROUP_KEY,SUBGROUP_VAL = get_arg_params()
    
    return INPUT_DIR,OUTPUT_DIR,MULTIPLOT,NEXPREPEAT,ROOT_INPUTVARS,SUBGROUP_KEY,SUBGROUP_VAL 


def setup_manually(NEXPREPEAT):
    ######          LOCAL SETUP        #######
    # perplot_params - one parameter per one plot
    PERPLOT_PARAMS= {'perplot_key':'subdomain_sizes','perplot_vals':['1','05','025','0125']} 
    UNFIXED_PARAM = 'input_vars_id'
    #TODO: PARAMS above just repeat each other?? 
    
    FIXED_PARAMS = {'tree_maxdepth':10,'eval_fraction':0.2,'regtypes':['random_forest'],'satdeficit':False}


    return PERPLOT_PARAMS, FIXED_PARAMS,UNFIXED_PARAM



class PlotDataProc():

    def __init__(
            self,
            INPUT_DIR):
        self.INPUT_DIR = INPUT_DIR


    @staticmethod
    def df_binary_split(sgval,sgkey):

        # below - boolean lists of df entries where the condition is/not satisfied
        with_sgval = [sgval in i for i in  subdf[f'{sgkey}'].values]
        without_sgval = [sgval not in i for i in  subdf[f'{sgkey}'].values]
        # slice using bool lists
        df_with_sgval = subdf.loc[with_sgval]
        df_without_sgval = subdf.loc[without_sgval]

        return df_with_sgval,df_without_sgval


    @staticmethod
    def get_readable_expid(unsorted_indices):
        '''
        DESCRIPTION
            sorting experiment id's to more readable form 
            from 'R3021' -> 'R0123', 
            from 'R102'  -> 'R012'
            where R - root input vars,integers - additional vars
        INPUT
            unsorted_indices - pandas series, containing
            index strings as in description
        '''

        sorted_indices = []
        for uindex in unsorted_indices:
            index_integers = sorted([int(i) for i in uindex.strip('R')])
            index_strings = [str(i) for i in index_integers]
            sorted_index = f'R{"".join(index_strings)}'
            sorted_indices.append(sorted_index)

        return sorted_indices


    def merge_dfs(self):
        '''
        DESCRIPTION
            reads all the CSV's within input dir
            returns df which merges the data of the output csv's within input_dir
        '''
        # dtypedict - columns data types for the Pandas DataFrame
        dtypedict= {'input_vars_id':str,'input_vars':str, 'satdeficit':np.bool_,'eval_fraction':np.float64,
                     'regtypes':str,'tree_maxdepth':np.int_,'subdomain_sizes':str,'refstd':np.float64,
                     'samplestd':np.float64,'samplecorr':np.float64,'exectime':str}
        # get list of CSV files from the input directory
        result_files = [file for file in os.listdir(input_dir) if "csv"  in file]
        df_list = []
        for resfile in result_files:
            fpath = os.path.join(self.INPUT_DIR,resfile)
            df = pd.read_csv(fpath,sep='\t',index_col=0,dtype=dtypedict)

            # take care of permutations of ID's
            first_id = df.iloc[0]['input_vars_id']
            df = df.assign(input_vars_id = first_id)

            df_list.append(df)

        df_merged = pd.concat(df_list)
        self.df_merged = df_merged 

        return self.df_merged


    def select_subdf(self,pkey,pval,fixed_params):

        perplot_df = self.df_merged.loc[self.df_merged[pkey] == pval]

        for fdkey,fdval in fixed_params.items():
            try:
                # assume it's an iterable
                subdf = perplot_df.loc[perplot_df[fdkey].isin(fdval)]
                # if fparval is not an iterable
            except TypeError:
                subdf = perplot_df.loc[perplot_df[fdkey] == fdval]
            perplot_df = subdf

        return subdf


    def singleplot_data(self,penv,pval,pkey):

        # SELECT DATA
        subdf = self.select_subdf(pkey,pval,self.fixed_params)

        subdf = subdf.sort_values(by='input_vars_id')
        # correct index
        npoints = subdf.shape[0]
        # npoints on a plot = n rows in the Pandas DF = n experiments
        subdf.index=list(range(npoints))
        # correct expid's 
        sorted_expids = PlotDataProc.sort_expids(subdf['input_vars_id'])
        subdf = subdf.assign(input_vars_id=sorted_expids)
        
        binary_split = False
        df_with = None
        df_without = None
        if ((sgval != None) and (sgkey! = None)):
            binary_split = True
            df_with,df_without = PlotDataProc.df_binary_split(subdf,sgval,sgkey)

        df_dict = {"subdf":subdf,"df_with":df_with,"df_without":df_without}

        return df_dict,binary_split


class PlotEnv():

    def __init__(
            self,
            INPUT_DIR,
            OUTPUT_DIR,
            PERPLOT_PARMS,
            FIXED_PARAMS,
            ROOT_INPUTVARS,
            NEXPREPEAT   = 0,
            SUBGROUP_KEY = None,
            SUBGROUP_VAL = None,
            MULTIPLOT    = False
        ):
        
        self.input_dir   = INPUT_DIR
        self.output_dir  = OUTPUT_DIR
        self.perplot_key = PERPLOT_PARAMS.pop("perplot_key","subdomain_sizes")
        self.perplot_vals = PERPLOT_PARAMS.pop("perplot_vals",['1'])
        self.tree_maxdepth = FIXED_PARAMS.pop("tree_maxdepth",10)
        self.eval_fraction = FIXED_PARAMS.pop("eval_fraction",0.2)
        self.regtypes      = FIXED_PARAMS.pop("regtypes",['random_forest'])
        self.satdeficit    = FIXED_PARAMS.pop("satdeficit",False)
        self.input_vars_id = FIXED_PARAMS.pop("input_vars_id",None)
        self.input_vars    = FIXED_PARAMS.pop("input_vars",None)
        if NEXPREPEAT!=0:
            #if repeating experiments, same input vars, no additional vars
            self.input_vars_id = FIXED_PARAMS.pop('input_vars_id','R')
            self.input_vars    = FIXED_PARAMS.pop('input_vars',f"['{ROOT_INPUTVARS}']")

        self.unfixed_param = UNFIXED_PARAM
        # root_inputvars="qtm,qsm,pm,tm"
        self.root_inputvars = ROOT_INPUTVARS
        self.nexprepeat     = NEXPREPEAT
        if NEXPREPEAT==0:
            self.repeated_exps = False
        else:
            self.repeated_exps = True
        self.subgroup_key   = SUBGROUP_KEY
        self.subgroup_val   = SUBGROUP_VAL
        self.multiplot      = MULTIPLOT


    def plotter(self,dproc):
        self.df_merged = dproc.merge_dfs()
        
        if self.multiplot is True:
            # plot multiple axes within one PNG
            # sefl.multiaxis_plot
            pass
        elif self.multiplot is False:
            self.monoaxis_plot(dproc)
        else:
            print("TypeError;multiplot value,should be bool!")


    def monoaxis_plot(self,dproc):
        '''
        DESCRIPTION
            Plot with one axes per plot, several .png files
        '''  
        pkey = self.perplot_key
        for pval in self.perplot_vals:
            sgval = self.subgroup_val
            sgkey = self.subgroup_key
            df_dict,binary_split = dproc.singleplot_data(pval,pkey,sgval,sgkey)
            self.generic_plot(df_dict,binary_split)


    def generic_plot(self,df_dict,binary_split):
        '''
        The logic of specifying the particular plots
        is hidden within the generic_plot intentionally.
        '''
        subdf = df_dict.pop('subdf',None)
        df_with = df_dict.pop('df_with',None)
        df_without = df_dict.pop('df_without',None)
        sorted_expids = subdf['input_vars_id']

        if self.repeated_exps:
            # plot T.diagram for the result of exps with REPEATED Invars
            self.repexp_plot(subdf,sorted_expids)
        else:
            # plot T.diagram for the result of exps with UNIQUE Invars
            self.uniqexp_plot(subdf,sorted_expids,binary_split)


    def repexp_plot(self,subdf,sorted_expids):
        '''
        DESCRIPTION
            plot unordered single sequence of points, representing
            multiple runs of the experiment with the same input_vars_id.
            Differences of exp results are thus only* due to randomness of 
            input datatset division
            * - probably some other factors influencing ML run
            outputs only 1 .PNG

        '''
        fig = plt.figure(figsize=(18,9))
        rootexpid = sorted_expids[0]


        refstd_mean = subdf.refstd.mean()
        ##########CREATE SEPARATE DIAGRAM-PICTURE##########
        # value of the reference standart deviation used only once per plot
        taylor_diagram = taylor(refstd_mean, fig=fig,# rect=111 by default anyway 
                                label='Reference')
        # dia.ax.plot(x95,y95,color='red')
        fig.tight_layout()

        # Add RMS contours, and label them
        contours = taylor_diagram.add_contours(levels=5, colors='0.5') # 5 levels
        taylor_diagram.ax.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

        ######### ITERATE OVER FIG POINTS(SAME EXP,DIFFERENT RUNS) ##################
        expgenerator = subdf.iterrows()
        for expindex,series in expgenerator:
            
            explabel  = str(series[self.unfixed_param][0]) # e.g. input_vars_id 
            expcorr  = series.samplecorr
            expstd = series.samplestd
            # add samples
            taylor_diagram.add_sample(expstd, expcorr,
                                   # ls - line style, supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed'
                                   # marker - how the point will be shown on the plot
                                   marker='.', ms=10, ls='',
                                   #mfc='k', mec='k', # B&W
                                   label=explabel)


    def uniqexp_plot(self,subdf,sorted_expids,binary_split):
        '''
        Further specify the type of plot
        mode1 = simply plot unique exp values
        mode2 = mode1 + divide the resulting points into
        2 groups based on some input var value.
        '''
        if binary_split:    
            self.uplot_binary(subdf,sorted_expids)
        else:
            self.uplot(subdf,sorted_expids)


    def uplot(self,subdf,sorted_expids):
        '''
        DESCRIPTION
            plot ordered multiple sequences of points, each representing
            how exp results change while adding extra variables
            outputs only 1 .PNG
        '''
        fig = plt.figure(figsize=(18,9))

        rootexpid = get_rootid(sorted_expids)
        expid_threads = get_threadset(rootexpid,sorted_expids)
        colors = plt.matplotlib.cm.brg(np.linspace(0,1,len(expid_threads)))
        # calculate references std as average of different runs (due to random splitting of dataset)
        refstd_mean = subdf.refstd.mean()

        ##########CREATE SEPARATE DIAGRAM-PICTURE##########
        # value of the reference standart deviation used only once per plot
        taylor_diagram = taylor(refstd_mean, fig=fig, rect=subplot_coords,
                                label='Reference')
        # dia.ax.plot(x95,y95,color='red')
        fig.tight_layout()


        ######### ITERATE OVER FIG POINTS(EXPERIMENTS) ##################
        for tindex,thread in enumerate(expid_threads):
            expstd_list = []
            expcorr_list = []
            for expid in thread:
                branch=subdf.loc[subdf['input_vars_id']==expid]     
                expstd_list.append(branch['samplestd'].values[0])
                expcorr_list.append(branch['samplecorr'].values[0])

            # plot threads, thread - scatter points connected by line
            taylor_diagram.add_sample_multimarkers(expstd_list, expcorr_list,tindex,
                                   # ls - line style, supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed'
                                   # marker - how the point will be shown on the plot
                                   color = colors[tindex],
                                   marker='', ms=10, ls='-',lw=1.0,
                                   label = '->'.join(thread))


    def uplot_binary(self,subdf,sorted_expids):
        '''
        DESCRIPTION
            plot ordered multiple sequences of points, each representing
            how exp results change while adding extra variables
            outputs only 1 .PNG
            Then plot 2 groups of points, 1st - including some input parameter(e.g. qlm),
            2nd group - excluding this paramter
        '''
        self.uplot()

        # add with QLM and without QLM
        std_with = df_with_sval['samplestd'].values
        corr_with = df_with_sval['samplecorr'].values
        taylor_diagram.add_sample(std_with, corr_with,
                               # ls - line style, supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed'
                               # marker - how the point will be shown on the plot
                               marker='.', ms=10, ls='',color="black",
                               #mfc='k', mec='k', # B&W
                               label=f"with {SUBGROUP_KEY}:{SUBGROUP_VAL}")

        std_without = df_without_sval['samplestd'].values
        corr_without = df_without_sval['samplecorr'].values
        taylor_diagram.add_sample(std_without, corr_without,
                               # ls - line style, supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed'
                               # marker - how the point will be shown on the plot
                               marker='.', ms=10, ls='',color="red",
                               #mfc='k', mec='k', # B&W
                               label=f"without {SUBGROUP_KEY}:{SUBGROUP_VAL}")
        # TODO: add function to put information boxes into the picture
        # TODO: and save the picture



def main():

    INPUT_DIR,OUTPUT_DIR,MULTIPLOT,NEXPREPEAT,ROOT_INPUTVARS,SUBGROUP_KEY,SUBGROUP_VAL = setup_from_CLI() 
    PERPLOT_PARAMS, FIXED_PARAMS,UNFIXED_PARAM = setup_manually()

    dproc = PlotDataProc(INPUT_DIR)
    penv = PlotEnv(INPUT_DIR,OUTPUT_DIR,PERPLOT_PARAMS,FIXED_PARAMS,UNFIXED_PARAM,ROOT_INPUTVARS,
                    NEXPREPEAT,SUBGROUP_KEY,SUBGROUP_VAL,MULTIPLOT)
    penv.plotter(dproc)

if __name__ == '__main__':

    main() 
