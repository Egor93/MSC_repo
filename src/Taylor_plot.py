import json 
import sys
import matplotlib.pyplot as plt
# Taylor Diagram from Yannick Copin <yannick.copin@laposte.net> 
taylor_path = './Taylor_Diagram' 
# sys.path.append(src_path)
sys.path += [taylor_path]
####################
#import cf_tree_regression as cft
from taylorDiagram import TaylorDiagram as taylor



with open('ML_performance_out.json','r') as f:
    jdict = dict(json.load(f))
    

subplot_coords = {'1' : 121, '05' : 122}

fig = plt.figure(figsize=(18,9))
fig.suptitle("Taylor diagram of ML cloud fraction simulations \n (max_depth_in=10) ", size='x-large')
for resolution in ['1','05']:

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
plt.savefig('../Img/Taylor_diag.png')
plt.show()
