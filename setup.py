#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, unicode_literals, absolute_import,
                        print_function)

import setuptools
from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'numpy',
    'matplotlib',
]

setup(
    name='ciabatta',
    version='0.6.0',
    description='Miscellaneous shared utilities',
    long_description=readme,
    author='Elliot Marsden',
    author_email='elliot.marsden@gmail.com',
    url='https://github.com/eddiejessup/ciabatta',
    packages=setuptools.find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    install_requires=requirements,
    license='BSD',
    keywords='ciabatta',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.6',
    ],
)
