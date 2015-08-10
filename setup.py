from numpy.distutils.core import setup, Extension
import numpy

setup(
    name="Shared libraries",
    author="Elliot Marsden",
    author_email="elliot.marsden@gmail.com",
    description="Useful miscellaneous functions and classes.",
    ext_modules=[
        Extension("field_numerics", ["field_numerics.pyx"],
                  include_dirs=[numpy.get_include()]),
        Extension("walled_field_numerics", ["walled_field_numerics.pyx"],
                  include_dirs=[numpy.get_include()]),
        Extension("geom_numerics", ["geom_numerics.pyx"],
                  include_dirs=[numpy.get_include()]),
        Extension("distance_numerics", ["distance_numerics.pyx"],
                  include_dirs=[numpy.get_include()]),
        Extension("lattice_numerics", ["lattice_numerics.pyx"],
                  include_dirs=[numpy.get_include()]),
        Extension("_periodic_cluster", sources=["periodic_cluster.f90"],
                  include_dirs=[numpy.get_include()]),
        Extension("cell_list._intro", sources=["cell_list/utils.f90",
                                               "cell_list/intro_shared.f90",
                                               "cell_list/intro_direct.f90",
                                               "cell_list/intro_2d.f90",
                                               "cell_list/intro_3d.f90"]),
    ]
)
