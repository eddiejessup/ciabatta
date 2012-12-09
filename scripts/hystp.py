#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as pp
import matplotlib.mlab as mlb
import numpy as np

def smooth(x, w=2):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < w:
        raise ValueError, "Input vector needs to be bigger than window size."
    s = np.r_[x[w-1:0:-1], x, x[-1:-w:-1]]
    c = np.ones(w, dtype=np.float)
    return np.convolve(c / c.sum(), s, mode='valid')

mpl.rc('font', family='serif', serif='Computer Modern Roman')
mpl.rc('text', usetex=True)

styles = ['r-', 'b-', 'g-', 'k-']

fnames = [
    '/home/ejm/Dropbox/Paper/Data/blank_rat_hyst_3/log.dat',
    '/home/ejm/Dropbox/Paper/Data/traps_1_rat_hyst_3/log.dat',
    '/home/ejm/Dropbox/Paper/Data/traps_5_rat_hyst_3/log.dat',
    '/home/ejm/Dropbox/Paper/Data/maze_rat_hyst_5/log.dat',
]

for i in range(len(fnames)):
    fname = fnames[i]
    n = np.loadtxt(fname).shape[1]
    if n == 4:
        r = mlb.csv2rec(fname, delimiter=' ', names=['time', 'dvar', 'frac', 'sense'])
    elif n ==3:
        r = mlb.csv2rec(fname, delimiter=' ', names=['time', 'dvar', 'sense'])
    else: raise Exception
    rs = smooth(np.asarray(r['sense'], dtype=np.float), 4)
    rd = smooth(np.asarray(r['dvar'], dtype=np.float), 4)
    pp.plot(rs, rd, styles[i], lw=0.8)

pp.xlabel(r'$\chi$, Chemotactic sensitivity', size=20)
pp.ylabel(r'$\sigma$, Spatial density standard deviation', size=20)
#pp.xlim([0.0, None])
#pp.ylim([0.0, None])
pp.show()
pp.savefig('hyst.png')


