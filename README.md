
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
       --highres /path/to/highres.nii \  # Required. Image defines inference space
       --stdhres /path/to/stdres.nii \   # Auxiliary data
       --covariance 0.806 0.131966 1 \   # [partial sill, bandwidth, exponent]
       --burnin 1000 \                   # MCMC burnin iterations
       --nsave 1000 \                    # MCMC iterations to save
       --thin 4 \                        # MCMC thinning factor
       --leapfrog 25 \                   # HMC number of integrator steps
       --mhtarget 0.65 \                 # HMC target acceptance rate
       --neighborhood 6.9 \              # Kriging approximation extent (mm)
       --threads 6 \                     # Number of cores to use
       --output output_basename \        # Output file base name
       --seed 8675309                    # URNG seed
```
