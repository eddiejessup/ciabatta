#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    'Cython',
    'numpy',
    'scipy',
    'matplotlib',
    'brewer2mpl',
]

test_requirements = [
    'Cython',
    'numpy',
    'scipy',
]

extensions = cythonize([
    Extension("ciabatta.distance_numerics",
              ["ciabatta/distance_numerics.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("ciabatta.field_numerics",
              ["ciabatta/field_numerics.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("ciabatta.geom_numerics",
              ["ciabatta/geom_numerics.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("ciabatta.lattice_numerics",
              ["ciabatta/lattice_numerics.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("ciabatta.walled_field_numerics",
              ["ciabatta/walled_field_numerics.pyx"],
              include_dirs=[numpy.get_include()]),
])

setup(
    name='ciabatta',
    version='0.2.2',
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
