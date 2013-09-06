from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name = "Shared libraries",
    author = "Elliot Marsden",
    author_email = "elliot.marsden@gmail.com",
    description = "Useful miscellaneous functions and classes.",
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension("field_numerics", ["field_numerics.pyx",]),
        Extension("walled_field_numerics", ["walled_field_numerics.pyx",]),
        Extension("geom_numerics", ["geom_numerics.pyx",]),
    ],
    include_dirs = [numpy.get_include()],
)
