// BSD 2-Clause License
//
// Copyright (c) 2023, Eijiro SHIBUSAWA
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>

namespace py = pybind11;

bool copyFromArray(const py::array &arr, unsigned long ptr)
{
	py::buffer_info buf = arr.request();
	int size = 1;
	for (int k = 0; k < buf.ndim; k++)
	{
		size *= buf.shape[k];
	}

	if (buf.format == py::format_descriptor<float>::format())
	{
		size *= sizeof(float);
	}
	else if (buf.format == py::format_descriptor<int>::format())
	{
		size *= sizeof(int);
	}
	else if (buf.format == py::format_descriptor<unsigned char>::format())
	{
		size *= sizeof(unsigned char);
	}
	else
	{
		std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
		return false;
	}
	char *ptr_ = reinterpret_cast<char *>(ptr);
	std::memcpy(ptr_, buf.ptr, size);

	return true;
}

bool copyToArray(py::array &arr, unsigned long ptr)
{
	py::buffer_info buf = arr.request();
	int size = 1;
	for (int k = 0; k < buf.ndim; k++)
	{
		size *= buf.shape[k];
	}

	if (buf.format == py::format_descriptor<float>::format())
	{
		size *= sizeof(float);
	}
	else if (buf.format == py::format_descriptor<int>::format())
	{
		size *= sizeof(int);
	}
	else if (buf.format == py::format_descriptor<unsigned char>::format())
	{
		size *= sizeof(unsigned char);
	}
	else
	{
		std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
		return false;
	}
	char *ptr_ = reinterpret_cast<char *>(ptr);
	std::memcpy(buf.ptr, ptr_, size);

	return true;
}

PYBIND11_MODULE(util_array_impl, m)
{
	m.def("copy_to_array", &copyToArray);
	m.def("copy_from_array", &copyFromArray);
}
