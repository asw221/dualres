
# Dual-resolution fMRI

#### Current dependencies
 - [boost](https://www.boost.org/)
 - [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
 - [FFTW3](http://www.fftw.org/)
 - [NLopt](https://nlopt.readthedocs.io/en/latest/)
 - [OpenMP](https://www.openmp.org/)
 - [zlib](https://www.zlib.net/)
 
 
#### Installation
Using cmake with dependencies installed:
```
cd dualres/lib/nifti && make all
mkdir ../../build && cd ../../build
cmake ..
make
```
 
#### To-do
 - Enhance comments and help pages
 - RBF parameter estimation: add option for fixed covariance
   parameters & toggle constraint
 - Convenient interface to check proportion of positive eigen values
   for given kernel function/image space
 - Add utility to convert image data types (double -> float) or vice
   versa
