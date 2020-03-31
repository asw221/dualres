
CXX := clang++
CXXFLAGS := -g -Wall -O0 -std=c++17

SRCDIR := src
BUILDDIR := build
EXECDIR := bin

## Depends on: boost, eigen3, fftw3, openmp, nlopt

INC := -Iinclude -Ilib/nifti/include -I/usr/local/include/eigen3 -I/usr/local/include
LIB := -Llib/nifti/lib -L/usr/local/lib -lniftiio -lznz -lz -lm -lnlopt -lboost_filesystem -lfftw3f -lfftw3f_omp -Xpreprocessor -fopenmp -lomp

dualresGP: $(SRCDIR)/dualresGP.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/dualresGP $<


estimateRbfParameters: $(SRCDIR)/estimateRbfParameters.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/estimateRbfParameters $<

preplan_fft: $(SRCDIR)/preplan_fft.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/preplan_fft $<

rbfNeighborhood: $(SRCDIR)/rbfNeighborhood.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/rbfNeighborhood $<
