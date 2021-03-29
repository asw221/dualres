
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(FFTW REQUIRED)
find_package(NLOPT REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)


if (NOT TARGET OpenMP::OpenMP_CXX)
  message(FATAL_ERROR "OpenMP_CXX target not found")
endif()

if (NOT TARGET FFTW::FloatOpenMP)
  message(FATAL_ERROR "FFTW float/omp target not found")
endif()
if (NOT TARGET FFTW::DoubleOpenMP)
  message(FATAL_ERROR "FFTW double/omp target not found")
endif()


# Anything that uses a CommandParser object must link to
# the nifti libraries



include_directories(${dualres_SOURCE_DIR}/include)
include_directories(${dualres_SOURCE_DIR}/lib/nifti/include)

link_directories(${dualres_SOURCE_DIR}/lib/nifti/lib)





## --- dualgpm -------------------------------------------------------
add_executable(dualgpm dualgpm.cpp)
target_link_libraries(dualgpm Boost::filesystem FFTW::FloatOpenMP
		FFTW::Float FFTW::DoubleOpenMP FFTW::Double
		Eigen3::Eigen NLOPT::nlopt OpenMP::OpenMP_CXX
		libniftiio.a libznz.a ZLIB::ZLIB)
set_target_properties(dualgpm PROPERTIES RUNTIME_OUTPUT_DIRECTORY
		"${CMAKE_BINARY_DIR}/bin")
target_compile_definitions(dualgpm PUBLIC DUALRES_DONT_UPDATE_TAU)
target_compile_definitions(dualgpm PUBLIC EIGEN_DONT_PARALLELIZE)

## EIGEN_DONT_PARALLELIZE
## EIGEN_DONT_VECTORIZE


add_executable(dualgpmf dualgpm.cpp)
target_link_libraries(dualgpmf Boost::filesystem FFTW::FloatOpenMP
		FFTW::Float FFTW::DoubleOpenMP FFTW::Double
		Eigen3::Eigen NLOPT::nlopt OpenMP::OpenMP_CXX
		libniftiio.a libznz.a ZLIB::ZLIB)
set_target_properties(dualgpmf PROPERTIES RUNTIME_OUTPUT_DIRECTORY
		"${CMAKE_BINARY_DIR}/bin")
target_compile_definitions(dualgpmf PUBLIC DUALRES_SINGLE_PRECISION
		DUALRES_DONT_UPDATE_TAU)
target_compile_definitions(dualgpmf PUBLIC EIGEN_DONT_PARALLELIZE)



## --- clear_fftw_history --------------------------------------------
add_executable(clear_fftw_history clear_fftw_history.cpp)
target_link_libraries(clear_fftw_history Boost::filesystem)
set_target_properties(clear_fftw_history PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")



## --- estimate_rbf --------------------------------------------------
add_executable(estimate_rbf estimate_rbf.cpp)
target_link_libraries(estimate_rbf Boost::filesystem Eigen3::Eigen
		NLOPT::nlopt libniftiio.a libznz.a ZLIB::ZLIB)
set_target_properties(estimate_rbf PROPERTIES RUNTIME_OUTPUT_DIRECTORY
		"${CMAKE_BINARY_DIR}/bin")


## --- estimate_snr --------------------------------------------------
add_executable(estimate_snr estimate_snr.cpp)
target_link_libraries(estimate_snr Boost::filesystem Eigen3::Eigen
		libniftiio.a libznz.a ZLIB::ZLIB)
set_target_properties(estimate_snr PROPERTIES RUNTIME_OUTPUT_DIRECTORY
		"${CMAKE_BINARY_DIR}/bin")


## --- gaussian_smooth -----------------------------------------------
add_executable(gaussian_smooth gaussian_smooth.cpp)
target_link_libraries(gaussian_smooth Boost::filesystem Eigen3::Eigen
		libniftiio.a libznz.a ZLIB::ZLIB)
set_target_properties(gaussian_smooth PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


## --- image_info ----------------------------------------------------
add_executable(image_info image_info.cpp)
target_link_libraries(image_info Boost::filesystem Eigen3::Eigen
		libniftiio.a libznz.a ZLIB::ZLIB)
set_target_properties(image_info PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


## --- interpolate ---------------------------------------------------
add_executable(interpolate interpolate.cpp)
target_link_libraries(interpolate Boost::filesystem Eigen3::Eigen
		Eigen3::Eigen OpenMP::OpenMP_CXX
		libniftiio.a libznz.a ZLIB::ZLIB)
set_target_properties(interpolate PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


## --- preplan_fft ---------------------------------------------------
add_executable(preplan_fft preplan_fft.cpp)
target_link_libraries(preplan_fft Boost::filesystem
		FFTW::Double FFTW::DoubleOpenMP
		FFTW::Float FFTW::FloatOpenMP
		Eigen3::Eigen OpenMP::OpenMP_CXX
		libniftiio.a libznz.a ZLIB::ZLIB)
set_target_properties(preplan_fft PROPERTIES RUNTIME_OUTPUT_DIRECTORY
		"${CMAKE_BINARY_DIR}/bin")


add_executable(preplan_fftf preplan_fft.cpp)
target_link_libraries(preplan_fftf Boost::filesystem 
		FFTW::Double FFTW::DoubleOpenMP
		FFTW::Float FFTW::FloatOpenMP
		Eigen3::Eigen OpenMP::OpenMP_CXX
		libniftiio.a libznz.a ZLIB::ZLIB)
set_target_properties(preplan_fftf PROPERTIES RUNTIME_OUTPUT_DIRECTORY
		"${CMAKE_BINARY_DIR}/bin")
target_compile_definitions(preplan_fftf
	PUBLIC DUALRES_SINGLE_PRECISION)



## --- rbf_neighborhood ----------------------------------------------
add_executable(rbf_neighborhood rbf_neighborhood.cpp)
target_link_libraries(rbf_neighborhood Boost::filesystem Eigen3::Eigen
		libniftiio.a libznz.a ZLIB::ZLIB)
set_target_properties(rbf_neighborhood PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")



## --- simulate_data_from_image --------------------------------------
add_executable(simulate_data_from_image simulate_data_from_image.cpp)
target_link_libraries(simulate_data_from_image Boost::filesystem
		Eigen3::Eigen libniftiio.a libznz.a ZLIB::ZLIB)
set_target_properties(simulate_data_from_image PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

