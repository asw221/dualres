
CXX := clang++
CXXFLAGS := -g -Wall -std=c++17 -O2 -DNDEBUG -DDUALRES_SINGLE_PRECISION

## -DDUALRES_SINGLE_PRECISION

SRCDIR := src
BUILDDIR := build
EXECDIR := bin

## Depends on: boost, eigen3, fftw3, openmp, nlopt
EIGEN3_DIR := /usr/local/include/eigen3

INC := -Iinclude -Ilib/nifti/include -I$(EIGEN3_DIR) -I/usr/local/include
LIB := -Llib/nifti/lib -L/usr/local/lib -lniftiio -lznz -lz -lnlopt -lboost_filesystem -lfftw3 -lfftw3_omp -lfftw3f -lfftw3f_omp  -Xpreprocessor -fopenmp -lomp -lm



all: niftilib dualgpm clear_fftw_history estimate_rbf estimate_snr gaussian_smooth image_info preplan_fft rbf_neighborhood simulate_data_from_image


niftilib:
	cd lib/nifti
	make clean
	make all
	echo "NIfTI lib build"
	cd ../../



dualgpm: $(SRCDIR)/dualgpm.cpp
	$(CXX) $< $(CXXFLAGS) $(INC) $(LIB) -DEIGEN_DONT_PARALLELIZE -DEIGEN_DONT_VECTORIZE -o $(EXECDIR)/dualgpm

clear_fftw_history: $(SRCDIR)/clear_fftw_history.cpp
	$(CXX) $< $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/clear_fftw_history

estimate_rbf: $(SRCDIR)/estimate_rbf.cpp
	$(CXX) $< $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/estimate_rbf

estimate_snr: $(SRCDIR)/estimate_snr.cpp
	$(CXX) $< $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/estimate_snr

gaussian_smooth: $(SRCDIR)/gaussian_smooth.cpp
	$(CXX) $< $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/gaussian_smooth

image_info: $(SRCDIR)/image_info.cpp
	$(CXX) $< $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/image_info

preplan_fft: $(SRCDIR)/preplan_fft.cpp
	$(CXX) $< $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/preplan_fft

rbf_neighborhood: $(SRCDIR)/rbf_neighborhood.cpp
	$(CXX) $< $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/rbf_neighborhood

simulate_data_from_image: $(SRCDIR)/simulate_data_from_image.cpp
	$(CXX) $< $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/simulate_data_from_image
