#!/bin/bash
f2py -c -m _intro ../Fortran/utils.f90 intro_shared.f90 intro_direct.f90 intro_2d.f90 intro_3d.f90
