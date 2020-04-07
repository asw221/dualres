
CXX := clang++
CXXFLAGS := -g -Wall -std=c++17 -O2

SRCDIR := src
BUILDDIR := build
EXECDIR := bin

## Depends on: boost, eigen3, fftw3, openmp, nlopt

INC := -Iinclude -Ilib/nifti/include -I/usr/local/include/eigen3 -I/usr/local/include
LIB := -Llib/nifti/lib -L/usr/local/lib -lniftiio -lznz -lz -lm -lnlopt -lboost_filesystem -lfftw3f -lfftw3f_omp -Xpreprocessor -fopenmp -lomp





dualgpm: $(SRCDIR)/dualgpm.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/dualgpm $<

clear_fftw_history: $(SRCDIR)/clear_fftw_history.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/clear_fftw_history $<

estimate_rbf: $(SRCDIR)/estimate_rbf.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/estimate_rbf $<

preplan_fft: $(SRCDIR)/preplan_fft.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/preplan_fft $<

rbf_neighborhood: $(SRCDIR)/rbf_neighborhood.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/rbf_neighborhood $<
