#!/usr/bin/env python

import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as pp
import fields

parser = argparse.ArgumentParser(description='Process a directory of states into a dvar data file')
parser.add_argument('-d', '--dir', default=os.getcwd(),
    help='state directory to be processed, defaults to cwd')
parser.add_argument('-s', '--silent', default=False, action='store_true',
    help='no output')
args = parser.parse_args()

if dir_name[-1] == '/': dir_name = dir_name.rstrip('/')
w_dat = np.load('%s/walls.npz' % dir_name)
walls, L = w_dat['walls'], w_dat['L']
valids = np.logical_not(np.asarray(walls, dtype=np.bool))
M = walls.shape[0]
dx = L / M
dim = walls.ndim

rs = sorted(glob.glob('%s/r/*.npz' % dir_name))
if len(rs) == 0: 
    raise Exception('Did not find any states')
f = open('%s/dvar.dat' % dir_name, 'w')
f.write('# dvar\n')
for i in range(len(rs)):
    r_dat = np.load(rs[i])
    r, t = r_dat['r'], r_dat['t']

    std = np.std(fields.density(r, L, dx)[valids])

    f.write('%f\n' % std)
    f.flush()
    if not args.silent: print('%.2f%%' % (100.0 * float(i) / len(rs)))
f.close()
