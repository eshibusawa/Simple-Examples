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

project(BoostPythonExample)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
	CACHE PATH "Output directory for libraries.")

# Python / Anaconda
set(PYTHON_VERSION 3.6)
string(REPLACE "." "" PYTHON_VERSION_REP ${PYTHON_VERSION})
set(ANACONDA_ROOT "/path/to/env/")

set(BOOST_PYTHON_LIBS python${PYTHON_VERSION}m boost_python${PYTHON_VERSION_REP} boost_numpy${PYTHON_VERSION_REP})
set(BOOST_PYTHON_INCLUDES ${ANACONDA_ROOT}/include ${ANACONDA_ROOT}/include/python${PYTHON_VERSION}m)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "${ANACONDA_ROOT}/lib/python3.6/site-packages" CACHE PATH "..." FORCE)
endif()
link_directories(${ANACONDA_ROOT}/lib)

add_library(simple_boost SHARED
	simple_boost.cpp
	)
target_include_directories(simple_boost PRIVATE ${BOOST_PYTHON_INCLUDES})
target_link_libraries(simple_boost ${BOOST_PYTHON_LIBS})
set_target_properties(simple_boost PROPERTIES PREFIX "")
install(TARGETS simple_boost DESTINATION .)
