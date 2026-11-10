from distutils.core import setup, Extension
import numpy.distutils.misc_util

# Adding OpenCV to project
# ************************

# Adding sources of the project
# *****************************

m_name = "voxel_height"

SOURCES = ["../cpp_utils/cloud/cloud.cpp",
           "height_calculate/calculation.cpp",
           "height_calculate/wrapper.cpp"]

module = Extension(m_name,
                   sources=SOURCES,
                   extra_compile_args=['-std=c++11',
                                       '-D_GLIBCXX_USE_CXX11_ABI=0'])

setup(
    name="voxel_height",
    version=0.1,
    ext_modules=[module],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs())
