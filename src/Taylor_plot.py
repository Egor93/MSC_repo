import json 
import sys
import numpy as np
import matplotlib.pyplot as plt
# Taylor Diagram from Yannick Copin <yannick.copin@laposte.net> 
taylor_path = './Taylor_Diagram' 
# sys.path.append(src_path)
sys.path += [taylor_path]
####################
#import cf_tree_regression as cft
from taylorDiagram import TaylorDiagram as taylor
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-f","--input_file",required = True, help = "input JSON file name")
parser.add_argument("-o","--output_file",required = True, help = "output PNG file")
args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file


def get_rows_cols(nplots):
    root = np.sqrt(nplots)

    if root.is_integer():
        rows = cols = root
    elif np.round(root) == np.ceil(root):
        rows = cols = np.ceil(root)
    else: 
        rows = np.floor(root)
        cols = rows + 1

    return int(rows),int(cols)

#with open('ML_performance_out1-0125.json','r') as f:
with open(input_file,'r') as f:
    jdict = dict(json.load(f))
    
nplots=len(jdict['references'].keys())
nrows, ncols = get_rows_cols(nplots)
plot_coords = [f'{nrows}{ncols}{i}' for i in range(1,nplots+1)]
plot_keys = list(jdict['references'].keys())
subplot_coords = dict(zip(plot_keys, plot_coords))
#subplot_coords = {'1' : 121, '05' : 122}

fig = plt.figure(figsize=(18,9))
fig.suptitle("Taylor diagram of ML cloud fraction simulations \n (max_depth_in=10) ", size='x-large')
for resolution in plot_keys:

    # calculate references std as average of different runs (due to random splitting of dataset)
    nruns = len(jdict['references'][resolution])
    refstd = sum([jdict['references'][resolution][i][0] for i in range(nruns)])/nruns
    #ref = jdict['references'][resolution][0][0]

    ##########PLOTTING##########
    dia = taylor(refstd, fig=fig, rect=subplot_coords[resolution],
                            label='Reference')
    # dia.ax.plot(x95,y95,color='red')
    fig.tight_layout()

    samples = jdict['samples']
    for i,(stddev,corrcoef,name) in enumerate(samples[resolution]):

        # add samples
        dia.add_sample(stddev, corrcoef,
                               marker='$%d$' % (i+1), ms=10, ls='',
                               #mfc='k', mec='k', # B&W
                               mfc='red', mec='red', # Colors
                               label=name)

    # Add RMS contours, and label them
    contours = dia.add_contours(levels=5, colors='0.5') # 5 levels
    dia.ax.clabel(contours, inline=1, fontsize=10, fmt='%.2f')
    # Tricky: ax is the polar ax (used for plots), _ax is the
    # container (used for layout)
    dia._ax.set_title(f'{resolution} degrees \n subdomain size')

fig.legend(dia.samplePoints,
           [ p.get_label() for p in dia.samplePoints ],
           numpoints=1, prop=dict(size='small'), loc='center')

fig.tight_layout()
plt.savefig(f'../Img/{output_file}.png')
plt.show()
