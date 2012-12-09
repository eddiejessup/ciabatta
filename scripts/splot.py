#!/usr/bin/env python

import os
import argparse
import subprocess
import glob
import numpy as np
import matplotlib.pyplot as pp

def out_nonint(fname):
    fname = '%s.png' % fname.rstrip('.npz')
    pp.savefig(fname)
    cmd = 'eog %s' % fname
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]

def out_int(fname):
    pp.show()

out = out_nonint

parser = argparse.ArgumentParser(description='Plot a box state')
parser.add_argument('i', type=int, default=-1, nargs='?',
    help='the identifier for the particular box state to plot')
parser.add_argument('-d', '--dir', default=os.getcwd(),
    help='the directory containing the box state, defaults to cwd')
args = parser.parse_args()

if args.dir[-1] == '/': args.dir = args.dir.rstrip('/')

w_dat = np.load('%s/walls.npz' % args.dir)
walls, L_half = w_dat['walls'], w_dat['L'] / 2.0

if args.i == -1: 
    fname = sorted(glob.glob('%s/r/*.npz' % args.dir))[-1]
else:
    rs = sorted(glob.glob('%s/r/%08i.npz' % (args.dir, args.i)))
    assert len(rs) == 1
    fname = rs[0]
r_dat = np.load(fname)
r, t = r_dat['r'], r_dat['t']

lims = [-L_half, L_half]
pp.imshow(walls.T, cmap='BuGn', interpolation='nearest', extent=2*lims, origin='lower')
pp.scatter(r[:, 0], r[:, 1], marker='.', s=0.1)
pp.figtext(0.5, 0.93, r't = %g s' % t, horizontalalignment='center')
pp.xlim(lims)
pp.ylim(lims)
pp.xticks([])
pp.yticks([])
out(fname.rstrip('.npz'))
