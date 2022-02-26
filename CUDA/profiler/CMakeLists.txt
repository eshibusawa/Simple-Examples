cmake_minimum_required(VERSION 3.18)

project(Profiler CUDA CXX)
set(CMAKE_VERBOSE_MAKEFILE 1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

add_executable(matrix_add_2D
	matrix_add_2D.cu
	)
install(TARGETS matrix_add_2D DESTINATION .)
