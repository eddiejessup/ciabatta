#!/bin/bash
f2py3.2 -c -m _intro ../Fortran/utils.f90 intro_2d.f90 intro_3d.f90 intro_direct.f90 --include-paths ../Fortran/
