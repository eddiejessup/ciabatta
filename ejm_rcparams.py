from __future__ import print_function, division
import numpy as np
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.backend_bases import register_backend
import brewer2mpl

golden_ratio = (np.sqrt(5) - 1.0) / 2.0
almost_black = '#262626'
almost_white = '#FEFEFA'

set2 = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors
reds = brewer2mpl.get_map('Reds', 'sequential', 3).mpl_colormap
red_blue = brewer2mpl.get_map('RdBu', 'diverging', 11).mpl_colormap
red_blue = mpl.cm.RdBu
reds = mpl.cm.Reds


def set_pretty_plots(use_latex=False, use_pgf=False):
    if use_pgf:
        register_backend('pdf', FigureCanvasPgf, 'pgf')
        rcParams['pgf.texsystem'] = 'pdflatex'

    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['STIXGeneral']
    preamble = [r'\usepackage{siunitx}',
                r'\usepackage{lmodern}',
                r'\usepackage{subdepth}'
                r'\usepackage[protrusion = true, expansion = true]{microtype}',
                ]
    rcParams['text.latex.preamble'] = preamble
    rcParams['pgf.preamble'] = preamble
    if use_latex:
        rcParams['text.usetex'] = True
    rcParams['text.latex.unicode'] = True

    rcParams['axes.color_cycle'] = set2
    rcParams['axes.edgecolor'] = almost_black
    rcParams['axes.labelcolor'] = almost_black
    rcParams['text.color'] = almost_black
    rcParams['grid.color'] = almost_black
    rcParams['legend.scatterpoints'] = 1
    rcParams['legend.frameon'] = False
    rcParams['lines.linewidth'] = 2.0
    # rcParams['image.aspect'] = 'equal'
    # rcParams['image.origin'] = 'lower'
    rcParams['image.interpolation'] = 'nearest'


def prettify_axes(*axs):
    for ax in axs:
        for spine in ax.spines:
            ax.spines[spine].set_linewidth(0.5)
            ax.spines[spine].set_color(almost_black)
        ax.xaxis.label.set_color(almost_black)
        ax.yaxis.label.set_color(almost_black)
        ax.title.set_color(almost_black)
        [i.set_color(almost_black) for i in ax.get_xticklabels()]
        ax.tick_params(axis='both', colors=almost_black)

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


def get_figsize(width=512, factor=0.6, ratio=golden_ratio):
    """Get width using \showthe\textwidth in latex file, then see the tex compile log.
    """
    fig_width_pt = width * factor
    inches_per_pt = 1.0 / 72.27
    fig_width_in = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * ratio   # figure height in inches
    figsize = [fig_width_in, fig_height_in]  # fig dims as a list
    return figsize
