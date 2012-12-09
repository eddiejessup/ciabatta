#!/usr/bin/env python

import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as pp
import clusters

parser = argparse.ArgumentParser(description='Process a directory of states into an nclust data file')
parser.add_argument('R', type=float,
    help='cluster cut-off radius')
parser.add_argument('-d', '--dir', default=os.getcwd(),
    help='state directory to be processed, defaults to cwd')
parser.add_argument('-s', '--silent', default=False, action='store_true',
    help='no output')
args = parser.parse_args()

if args.dir[-1] == '/': args.dir = args.dir.rstrip('/')
L = np.load('%s/walls.npz' % args.dir)['L']

rs = sorted(glob.glob('%s/r/*.npz' % args.dir))
if len(rs) == 0: 
    raise Exception('Did not find any states')
f = open('%s/clust_%g.dat' % (args.dir, args.R), 'w')
f.write('# pclust_%g\n' % (args.R))
for i in range(len(rs)):
    r_dat = np.load(rs[i])
    r, t = r_dat['r'], r_dat['t']

    inds = clusters.get_inds(r, args.R, L)
    pclust = r.shape[0] / float(len(inds))

    f.write('%f\n' % (pclust))
    f.flush()
    if not args.silent: print('%.2f%%' % (100.0 * float(i) / len(rs)))
f.close()
