import numpy as np
import matplotlib
from matplotlib import rcParams
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 10
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
preamble = [r'\usepackage{siunitx}']
rcParams['text.latex.preamble'] = preamble
rcParams['pgf.preamble'] = preamble

def get_figsize(width=512, factor=0.6):
	fig_width_pt = width * factor
	inches_per_pt = 1.0 / 72.27
	golden_ratio = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
	fig_width_in = fig_width_pt * inches_per_pt  # figure width in inches
	fig_height_in = fig_width_in * golden_ratio   # figure height in inches
	figsize = [fig_width_in, fig_height_in] # fig dims as a list
	return figsize