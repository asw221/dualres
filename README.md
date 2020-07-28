
# Dual-resolution fMRI

#### Dependencies
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
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
 
#### To-do
 - [x] RBF parameter estimation: add option for fixed covariance
   parameters & toggle constraint
 - [ ] Convenient interface to check proportion of positive eigen values
   for given kernel function/image space
 - [ ] Enhance comments and help pages
 - [ ] Add options & utilities for masking images
 - [ ] Would eventually like to write wrapping classes for `nifti_image`
   pointers and covariance parameters



### Analysis
```
$ ./dualres/build/bin/dualgpmf \
       --highres /path/to/highres.nii \
       --stdhres /path/to/stdres.nii \
       --covariance 0.806 0.131966 1 \
       --burnin 1000 \
       --nsave 1000 \
       --thin 4 \
       --leapfrog 25 \
       --mhtarget 0.65 \
       --neighborhood 6.9 \
       --threads 6 \
       --output output_basename \
       --seed 8675309
```
