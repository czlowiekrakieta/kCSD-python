from distutils.core import setup
# from Cython.Build import cythonize
# from Cython.Distutils import Extension
# import numpy as np

# ext = Extension(name="coordinate_descent" ,sources=["kcsd/cythonized/coordinate_descent.pyx"], libraries=["gsl", "gslcblas"])

setup(name="kcsd",
      version="1.1",
      packages=["kcsd"]),
      # ext_modules=cythonize(ext),
      # include_dirs=[np.get_include()])