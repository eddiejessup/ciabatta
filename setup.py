#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, unicode_literals, absolute_import,
                        print_function)

import setuptools
from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    'numpy',
    'matplotlib',
    'brewer2mpl',
]

test_requirements = [
    'numpy',
]

setup(
    name='ciabatta',
    version='0.5.1',
    description='Miscellaneous shared utilities',
    long_description=readme + '\n\n' + history,
    author='Elliot Marsden',
    author_email='elliot.marsden@gmail.com',
    url='https://github.com/eddiejessup/ciabatta',
    packages=setuptools.find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    install_requires=requirements,
    license='BSD',
    zip_safe=False,
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
    ],
    test_suite='tests',
    tests_require=test_requirements,
)
