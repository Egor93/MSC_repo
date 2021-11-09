import json 
import numpy as np
import matplotlib.pyplot as plt
import argparse
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
        rows = np.floor(root)
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


def singleplot_image(nplots, jdict, output_file, plot_keys,colors):
    '''
    Make a separate .png diagram for each resolution
    '''
    subplot_coords = 111 # one subplot per .png

    fig = plt.figure(figsize=(18,9))
    for resolution in plot_keys:

        # calculate references std as average of different runs (due to random splitting of dataset)
        nruns = len(jdict['references'][resolution])
        refstd = sum([jdict['references'][resolution][i][0] for i in range(nruns)])/nruns
        #ref = jdict['references'][resolution][0][0]

        ##########CREATE SEPARATE DIAGRAM-PICTURE##########
        # value of the reference standart deviation used only once per plot
        taylor_diagram = taylor(refstd, fig=fig, rect=subplot_coords,
                                label='Reference')
        # dia.ax.plot(x95,y95,color='red')
        fig.tight_layout()

        # WITHIN EACH SEPARATE PICTURE
        # TODO : df.loc[df['column_name'] == some_value]
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
        taylor_diagram._ax.set_title(f'Taylor diagram of ML cloud fraction simulations \n (max_depth_in=10) \n\
                             {resolution} degrees \n subdomain size')
        

        fig.legend(taylor_diagram.samplePoints,
                   [ p.get_label() for p in taylor_diagram.samplePoints ],
                   numpoints=1, prop=dict(size='small'), loc='center')

        fig.tight_layout()
        # full filepath is provided in the main.sh call
        plt.savefig(f'{output_file}{resolution}.png')
        # plt.clf() clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.clf()


def main():
    # parse input variables
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--input_file",required = True, help = "input JSON file name")
    parser.add_argument("-o","--output_file",required = True, help = "output PNG file")
    # there are two possible output types: singleplot or multiplot
    parser.add_argument("-t","--output_type",required = True, help = "output several PNG files, one file per Taylor diagram or single plot")
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    output_type = args.output_type 
    
    #with open('ML_performance_out1-0125.json','r') as f:
    with open(input_file,'r') as f:
        jdict = dict(json.load(f))
    
    plot_keys = list(jdict['references'].keys())
    nplots = len(plot_keys)
    ncolors = len(jdict['references'][plot_keys[0]])
    # Colormap (see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps)
    colors = plt.matplotlib.cm.Set1(np.linspace(0,1,ncolors))

    if output_type == 'multiplot':
        multiplot_image(nplots,jdict,output_file,plot_keys,colors)
    elif output_type == 'singleplot':
        singleplot_image(nplots,jdict,output_file,plot_keys,colors)
    else:
        print("Error of defining output_type, try again!")

if __name__ == '__main__':

    main() 
