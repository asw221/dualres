
project(dualres)
cmake_minimum_required(VERSION 3.12)


set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")



add_subdirectory(src)



if (APPLE)
   set(CMAKE_C_COMPILER clang)
   set(CMAKE_CXX_COMPILER clang++)

   if (CMAKE_C_COMPILER_ID MATCHES "Clang\$")
      set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
      set(OpenMP_C_LIB_NAMES "omp")
      set(OpenMP_omp_LIBRARY omp)
   endif()

   if (CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
      set(OpenMP_CXX_LIB_NAMES "omp")
      set(OpenMP_omp_LIBRARY omp)
   endif()

endif()



