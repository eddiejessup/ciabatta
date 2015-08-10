#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools
from numpy.distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

cython_extensions = cythonize([
    Extension("_distance_numerics", ["ciabatta/distance_numerics.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("_field_numerics", ["ciabatta/field_numerics.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("_geom_numerics", ["ciabatta/geom_numerics.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("_lattice_numerics", ["ciabatta/lattice_numerics.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("_walled_field_numerics", ["ciabatta/walled_field_numerics.pyx"],
              include_dirs=[numpy.get_include()]),
])

cell_list_sources = ["ciabatta/cell_list_numerics/utils.f90",
                     "ciabatta/cell_list_numerics/cell_list_shared.f90",
                     "ciabatta/cell_list_numerics/cell_list_direct.f90",
                     "ciabatta/cell_list_numerics/cell_list_2d.f90",
                     "ciabatta/cell_list_numerics/cell_list_3d.f90"
                     ]

fortran_extensions = [
    Extension("ciabatta._periodic_cluster",
              sources=["ciabatta/periodic_cluster.f90"]),
    Extension("ciabatta._cell_list",
              sources=cell_list_sources),
]

extensions = cython_extensions + fortran_extensions

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    'brewer2mpl>=1.4.1',
    'Cython>=0.23',
    'matplotlib>=1.4.3',
    'numpy>=1.9.2',
    'scipy>=0.16.0',
]

test_requirements = [
    'numpy>=1.9.2',
    'scipy>=0.16.0',
]

setup(
    name='ciabatta',
    version='0.1.0',
    description="Miscellaneous shared utilities",
    long_description=readme + '\n\n' + history,
    author="Elliot Marsden",
    author_email='elliot.marsden@gmail.com',
    url='https://github.com/eddiejessup/ciabatta',
    packages=setuptools.find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='ciabatta',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    ext_modules=extensions,
)
