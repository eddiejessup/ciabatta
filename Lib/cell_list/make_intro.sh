#!/bin/bash
f2py3.2 -c -m _intro ../Fortran/utils.f90 intro.f90 --include-paths ../Fortran/
