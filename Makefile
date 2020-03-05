
CXX := clang++
CXXFLAGS := -g -Wall -O0 -std=c++17

SRCDIR := src
BUILDDIR := build
EXECDIR := bin

## Depends on: arrayfire, eigen3, nlopt

INC := -Iinclude -Ilib/nifti/include -I/usr/local/include/eigen3 -I/usr/local/include
LIB := -Llib/nifti/lib -L/usr/local/lib -lniftiio -lznz -lz -lafcpu -lm -lnlopt

dualresGP: $(SRCDIR)/dualresGP.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/dualresGP $<


estimateRbfParameters: $(SRCDIR)/estimateRbfParameters.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/estimateRbfParameters $<


rbfNeighborhood: $(SRCDIR)/rbfNeighborhood.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) -o $(EXECDIR)/rbfNeighborhood $<
