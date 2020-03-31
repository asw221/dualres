
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(FFTW REQUIRED)
find_package(NLOPT REQUIRED)
find_package(OpenMP REQUIRED)


if (NOT TARGET OpenMP::OpenMP_CXX)
  message(FATAL_ERROR "OpenMP_CXX target not found")
endif()


include_directories(${dualres_SOURCE_DIR}/include)
include_directories(${dualres_SOURCE_DIR}/lib/nifti/include)

link_directories(${dualres_SOURCE_DIR}/lib/nifti/lib)

## --- preplan_fft ---------------------------------------------------
add_executable(preplan_fft preplan_fft.cpp)
target_link_libraries(preplan_fft Boost::filesystem FFTW::FloatOpenMP FFTW::Float
		Eigen3::Eigen OpenMP::OpenMP_CXX libniftiio.a libznz.a
		libz.dylib)
set_target_properties(preplan_fft PROPERTIES RUNTIME_OUTPUT_DIRECTORY
		"${CMAKE_BINARY_DIR}/bin")

