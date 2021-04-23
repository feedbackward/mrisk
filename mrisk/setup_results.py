'''Setup: basic config for visualization of experimental results.'''

## External modules.
import matplotlib.pyplot as plt
import numpy as np


###############################################################################


img_dir = "img"

results_dir = "results"

my_fontsize = "xx-large"
my_ext = "pdf"


def export_legend(legend, filename="legend.pdf", expand=[-5,-5,5,5]):
    '''
    Save just the legend.
    Source for this: https://stackoverflow.com/a/47749903
    '''
    fig  = legend.figure
    fig.canvas.draw()
    plt.axis('off')
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


###############################################################################
