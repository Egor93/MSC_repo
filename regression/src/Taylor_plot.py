import json 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
# Taylor Diagram from Yannick Copin <yannick.copin@laposte.net> 
from external.diagram.taylorDiagram import TaylorDiagram as taylor


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


def multiplot_image(nplots, jdict, output_file, plot_keys):
    '''
    All Taylor diagrams in one .png file
    '''
    nrows, ncols = get_rows_cols(nplots)
    plot_coords = [f'{nrows}{ncols}{i}' for i in range(1,nplots+1)]
    subplot_coords = dict(zip(plot_keys, plot_coords))
    #subplot_coords = {'1' : 121, '05' : 122} required by taylor diagram method

    fig = plt.figure(figsize=(18,9))
    fig.suptitle("Taylor diagram of ML cloud fraction simulations \n (max_depth_in=10) ", size='x-large')
    for resolution in plot_keys:

        # calculate references std as average of different runs (due to random splitting of dataset)
        nruns = len(jdict['references'][resolution])
        refstd = sum([jdict['references'][resolution][i][0] for i in range(nruns)])/nruns
        #ref = jdict['references'][resolution][0][0]

        ##########PLOTTING##########
        taylor_diagram = taylor(refstd, fig=fig, rect=subplot_coords[resolution],
                                label='Reference')
        # taylor_diagram.ax.plot(x95,y95,color='red')
        fig.tight_layout()

        samples = jdict['samples']
        for i,(stddev,corrcoef,name) in enumerate(samples[resolution]):

            # add samples
            taylor_diagram.add_sample(stddev, corrcoef,
                                   marker='$%d$' % (i+1), ms=10, ls='',
                                   #mfc='k', mec='k', # B&W
                                   mfc=colors[i], mec=colors[i], # Colors
                                   label=name)

        # Add RMS contours, and label them
        contours = taylor_diagram.add_contours(levels=5, colors='0.5') # 5 levels
        taylor_diagram.ax.clabel(contours, inline=1, fontsize=10, fmt='%.2f')
        # Tricky: ax is the polar ax (used for plots), _ax is the
        # container (used for layout)
        taylor_diagram._ax.set_title(f'{resolution} degrees \n subdomain size')

    fig.legend(taylor_diagram.samplePoints,
               [ p.get_label() for p in taylor_diagram.samplePoints ],
               numpoints=1, prop=dict(size='small'), loc='center')

    fig.tight_layout()
    # full filepath is provided in the main.sh call
    plt.savefig(f'{output_file}.png')
    plt.show()


def singleplot_image(df_merged,fixed_params,perplot_params,unfixed_params,output_dir):
    '''
    Make a separate .png diagram for each resolution
    '''
    subplot_coords = 111 # one subplot per .png

    fig = plt.figure(figsize=(18,9))

    # select unique perplot parameter, e.g. 'subdomain_sizes'
    perplot_key = list(perplot_params.keys())[0]
    perplot_vals=perplot_params[perplot_key]
    for perplot_val in perplot_vals:

        # SELECT DATA
        subdf = select_subdf(df_merged,perplot_key,perplot_val,fixed_params)
        subdf = subdf.sort_values(by=unfixed_params)
        # npoints on a plot = n rows in the Pandas DF = n experiments
        npoints = subdf.shape[0]
        # correct index
        subdf.index=list(range(npoints))
        colors = plt.matplotlib.cm.brg(np.linspace(0,1,npoints))
        # calculate references std as average of different runs (due to random splitting of dataset)
        refstd_mean = subdf.refstd.mean()

        ##########CREATE SEPARATE DIAGRAM-PICTURE##########
        # value of the reference standart deviation used only once per plot
        taylor_diagram = taylor(refstd_mean, fig=fig, rect=subplot_coords,
                                label='Reference')
        # dia.ax.plot(x95,y95,color='red')
        fig.tight_layout()

        ######### ITERATE OVER FIG POINTS(EXPERIMENTS) ##################
        expgenerator = subdf.iterrows()
        for expindex,series in expgenerator:
            
            explabel  = str(series[unfixed_params[0]]) # e.g. input_vars_id 
            expcorr  = series.samplecorr
            expstd = series.samplestd
            # add samples
            taylor_diagram.add_sample(expstd, expcorr,
                                   # ls - line style, supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed'
                                   # marker - how the point will be shown on the plot
                                   marker='$%d$' % (expindex+1), ms=10, ls='',
                                   #mfc='k', mec='k', # B&W
                                   mfc=colors[expindex], mec=colors[expindex], # Colors
                                   label=explabel)

            # Add RMS contours, and label them
        contours = taylor_diagram.add_contours(levels=5, colors='0.5') # 5 levels
        taylor_diagram.ax.clabel(contours, inline=1, fontsize=10, fmt='%.2f')
        # Tricky: ax is the polar ax (used for plots), _ax is the
        # container (used for layout)

        perplot_str = f'perplot parameters:{perplot_key}={perplot_val}'
        fixed_str   = f'fixed parameters {fixed_params}'
        title = f'{perplot_str} \n {fixed_str}'
        taylor_diagram._ax.set_title(title)
                                    
        

        fig.legend(taylor_diagram.samplePoints,
                   [ p.get_label() for p in taylor_diagram.samplePoints ],
                   numpoints=1, prop=dict(size='small'), loc='lower right')

        fig.tight_layout()
        # full filepath is provided in the main.sh call
        fname = f'{perplot_key}_{perplot_val}.png'
        fpath = os.path.join(output_dir,fname)
        plt.savefig(fpath)
        # plt.clf() clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.clf()


def get_df_merged(input_dir,dtypedict):
    '''
    DESCRIPTION
        reads all the CSV's within input dir
        returns df which merges the data of the output csv's within input_dir
    '''
    result_files = os.listdir(input_dir) 
    df_list = []
    for resfile in result_files:
        fpath = os.path.join(input_dir,resfile)
        df = pd.read_csv(fpath,sep='\t',index_col=0,dtype=dtypedict)

        # take care of permutations of ID's
        first_id = df.iloc[0]['input_vars_id']
        df = df.assign(input_vars_id = first_id)

        df_list.append(df)

    df_merged = pd.concat(df_list)
    

    return df_merged


def select_subdf(df_merged,perplot_key,perplot_val,fixed_params):

    perplot_df = df_merged.loc[df_merged[perplot_key] == perplot_val]

    for fparkey,fparval in fixed_params.items():
        try:
            # assume it's an iterable
            subdf = perplot_df.loc[perplot_df[fparkey].isin(fparval)]
            # if fparval is not an iterable
        except TypeError:
            subdf = perplot_df.loc[perplot_df[fparkey] == fparval]
        perplot_df = subdf

    return subdf


def main():
    # parse input variables
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_dir",required = True, help = "input JSON file name")
    parser.add_argument("-o","--output_dir",required = True, help = "output PNG file")
    # there are two possible output types: singleplot or multiplot
    parser.add_argument("-m","--multiplot",required = True, help = "vals=True/False; output several PNG files, one file per Taylor diagram or single plot")
    args = parser.parse_args()
    # input_dir where results of experiments are stored, e.g. as CSV
    input_dir = args.input_dir
    output_dir = args.output_dir
    multiplot = eval(args.multiplot)
    

    #with open(input_file,'r') as f:
    #    jdict = dict(json.load(f))
    
    #result_cols = ['input_vars_id','input_vars','satdeficit','eval_fraction','regtypes','tree_maxdepth','subdomain_sizes',
    #                        'refstd','samplestd','samplevar','exectime']
    # perplot_params - one parameter per one plot
    dtypedict= {'input_vars_id':str,'input_vars':str, 'satdeficit':np.bool_,'eval_fraction':np.float64,
                 'regtypes':str,'tree_maxdepth':np.int_,'subdomain_sizes':str,'refstd':np.float64,
                 'samplestd':np.float64,'samplecorr':np.float64,'exectime':str}
    df_merged = get_df_merged(input_dir,dtypedict)


    perplot_params= {'subdomain_sizes':['1','05','025']} 
    assert len(perplot_params)==1, "should choose only one such parameter!"
    # fixed parameters within one plot
    fixed_params = {'tree_maxdepth':10,'eval_fraction':0.2,'regtypes':['decision_tree'],'satdeficit':False}
    # varying parameters within one plot
    # TODO: Is it necessary to explicitly define unfixed parameters?
    # TODO: Can be defined instead as the rest of the parameters.
    unfixed_params = ['input_vars_id']

    if multiplot is True:
        nplots = len(plot_keys)
        #multiplot_image(nplots,jdict,output_file,plot_keys,colors)
    elif multiplot is False:
        singleplot_image(df_merged,fixed_params,perplot_params,unfixed_params,output_dir)
    else:
        print("Error of defining output_type, try again!")

if __name__ == '__main__':

    main() 
