# BSD 2-Clause License
#
# Copyright (c) 2021, Eijiro SHIBUSAWA
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.3)

project(MexExample)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_FLAG_NATIVE_SUPPORTED)
if(COMPILER_FLAG_NATIVE_SUPPORTED)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

find_package(Eigen3 REQUIRED eigen3)
set(EIGEN3_FOUND "")
if(EIGEN3_INCLUDE_DIR)
	message("   Eigen3 enabled")
endif()

find_package(PkgConfig)
pkg_check_modules(OCTAVE REQUIRED octave)
if(OCTAVE_INCLUDE_DIRS)
	message("   Octave enabled")
endif()

set(OCT_COMPILER mkoctfile)
set(OCT_COMPILER_ARGS --mex)
set(OCT_CXX_FLAGS ${CMAKE_CXX_FLAGS})
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
	set(OCT_COMPILER_ARGS ${OCT_COMPILER_ARGS} -g)
	set(OCT_CXX_FLAGS "${OCT_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
else()
	set(OCT_CXX_FLAGS "${OCT_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
endif()
if(${CMAKE_VERBOSE_MAKEFILE})
	set(OCT_COMPILER_ARGS ${OCT_COMPILER_ARGS} -v)
endif()

add_custom_command(OUTPUT mexSimpleMM.mex
	COMMAND ${CMAKE_COMMAND} -E env CXXFLAGS=${OCT_CXX_FLAGS} ${OCT_COMPILER} ${OCT_COMPILER_ARGS} -I${EIGEN3_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/mexSimpleMM.cpp
	COMMAND ${CMAKE_COMMAND} -E copy mexSimpleMM.mex ${PROJECT_SOURCE_DIR}/
	WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/
)

add_custom_target(mexall ALL DEPENDS mexSimpleMM.mex)
