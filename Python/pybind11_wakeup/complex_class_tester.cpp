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

#define private public
#include "complex_class.hpp"
#undef private

#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class ComplexClassTester
{
private:
public:
	ComplexClassTester():
		m_debugger(m_cc.m_debugger)
	{
	}

	~ComplexClassTester()
	{
		if (m_worker.joinable())
		{
			m_debugger.stop();
			m_worker.join();
		}
	}

	void incrementInternalState()
	{
		m_debugger.incrementInternalState();
	}

	void resume()
	{
		m_debugger.resume();
	}

	int getInternalState()
	{
		std::vector<void *> d;
		m_debugger.get(0, d);
		int ret = *(reinterpret_cast<int *>(d[0]));
		return ret;
	}

	void setInternalState(int value)
	{
		std::vector<void *> d;
		m_debugger.get(0, d);
		*(reinterpret_cast<int *>(d[0])) = value;
	}

	void startWorker(int numSteps)
	{
		if (!m_debugger.m_stop)
		{
			return;
		}
		m_debugger.clear();
		m_debugger.m_enable = true;
		m_debugger.m_stop = false;
		m_worker = std::thread([this, numSteps]{m_cc.complexMethod(numSteps);});
		while (!m_worker.joinable())
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(m_debugger.m_waitMs));
		}
	}

private:
	ComplexClass m_cc;
	MTDebugger &m_debugger;
	std::thread m_worker;
};


PYBIND11_MODULE(complex_class, m)
{
	py::class_<ComplexClassTester> complex_class_tester(m, "complex_class_tester");

	complex_class_tester.def(py::init<>())
		.def("get_internal_state", &ComplexClassTester::getInternalState)
		.def("set_internal_state", &ComplexClassTester::setInternalState)
		.def("start_worker", &ComplexClassTester::startWorker)
		.def("increment_internal_state", &ComplexClassTester::incrementInternalState)
		.def("resume", &ComplexClassTester::resume)
		;
}
