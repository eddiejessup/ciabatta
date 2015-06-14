from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name="Shared libraries",
    author="Elliot Marsden",
    author_email="elliot.marsden@gmail.com",
    description="Useful miscellaneous functions and classes.",
    ext_modules=cythonize([
        Extension("field_numerics", ["field_numerics.pyx", ]),
        Extension("walled_field_numerics", ["walled_field_numerics.pyx", ]),
        Extension("geom_numerics", ["geom_numerics.pyx", ]),
        Extension("distance_numerics", ["distance_numerics.pyx", ]),
        Extension("lattice_numerics", ["lattice_numerics.pyx", ]),
    ]),
    include_dirs=[numpy.get_include()],
)
