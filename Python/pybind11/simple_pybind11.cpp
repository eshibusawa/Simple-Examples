// BSD 2-Clause License
//
// Copyright (c) 2022, Eijiro SHIBUSAWA
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

class SimpleClass
{
public:
	enum ComputationMode {Add, Sub, Mul, Div};

public:
	SimpleClass()
	{
	}

	py::object compute(const py::object &obj, const py::array &arr)
	{
		py::buffer_info buf = arr.request();
		if ((buf.ndim != 2) || (buf.strides[0] != buf.shape[1] * sizeof(float)) ||
			(buf.format != py::format_descriptor<float>::format()))
		{
			throw std::runtime_error("only C-contiguous 2D float array is supported");
		}

		auto r = arr.unchecked<float, 2>();
		auto result = py::array_t<float, py::array::c_style>({r.shape(0), r.shape(1)});

		const float value = obj.attr("value").cast<float>();
		ComputationMode m = obj.attr("computation_mode").cast<SimpleClass::ComputationMode>();
		auto rr = result.mutable_unchecked<2>();
		switch(m)
		{
			case Add:
				for (py::ssize_t j = 0; j < rr.shape(0); j++)
				{
					for (py::ssize_t i = 0; i < rr.shape(1); i++)
					{
						rr(j, i) = r(j, i) + value;
					}
				}
			break;

			case Sub:
				for (py::ssize_t j = 0; j < rr.shape(0); j++)
				{
					for (py::ssize_t i = 0; i < rr.shape(1); i++)
					{
						rr(j, i) = r(j, i) - value;
					}
				}
			break;

			case Mul:
				for (py::ssize_t j = 0; j < rr.shape(0); j++)
				{
					for (py::ssize_t i = 0; i < rr.shape(1); i++)
					{
						rr(j, i) = r(j, i) * value;
					}
				}
			break;

			case Div:
				for (py::ssize_t j = 0; j < rr.shape(0); j++)
				{
					for (py::ssize_t i = 0; i < rr.shape(1); i++)
					{
						rr(j, i) = r(j, i) / value;
					}
				}
			break;

			default:
				std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
				throw std::runtime_error("unknown computation mode");
			break;
		}

		return result;
	}

};

py::object get_class_a()
{
	py::module simple_class_main = py::module_::import("simple_class_main");
	return simple_class_main.attr("class_a")();
}

py::object get_class_b()
{
	py::module simple_class_main = py::module_::import("simple_class_main");
	return simple_class_main.attr("class_b")();
}

PYBIND11_MODULE(simple_pybind, m)
{
	py::class_<SimpleClass> simple_class(m, "simple_class");

	simple_class.def(py::init<>())
		.def("compute", &SimpleClass::compute);

	py::enum_<SimpleClass::ComputationMode>(simple_class, "computation_mode")
		.value("add", SimpleClass::Add)
		.value("sub", SimpleClass::Sub)
		.value("mul", SimpleClass::Mul)
		.value("div", SimpleClass::Div)
		.export_values();

	m.def("get_class_a", &get_class_a);
	m.def("get_class_b", &get_class_b);
}
