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

#include "complex_class.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <future>
#include <mutex>
#include <thread>

namespace py = pybind11;

class ComplexClassTester
{
private:
	static const int m_waitMs;
public:
	ComplexClassTester():
		m_previousState(-1)
	{
	}

	~ComplexClassTester()
	{
		if (m_worker.joinable())
		{
			m_cc.m_stop = true;
			m_cc.m_cv.notify_all();
			std::this_thread::sleep_for(std::chrono::milliseconds(m_waitMs));
			m_worker.join();
		}
	}

	int getInternalState()
	{
		int ret = m_previousState;
		do
		{
			{
				std::lock_guard<std::mutex> lk(m_cc.m_mutex);
				ret = m_cc.m_internalState;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(m_waitMs)); // loop wait, for simplicity
		} while (ret == m_previousState);
		m_cc.setEvent(ret);
		m_previousState = ret;
		return ret;
	}

	void startWorker()
	{
		if (!m_cc.m_stop)
		{
			return;
		}
		m_cc.m_stop = false;
		m_worker = std::thread([this]{m_cc.complexMethod();});
		while (!m_worker.joinable())
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(m_waitMs));
		}
	}

private:
	ComplexClass m_cc;
	std::thread m_worker;
	int m_previousState;
};

const int ComplexClassTester::m_waitMs = 10;

PYBIND11_MODULE(complex_class, m)
{
	py::class_<ComplexClassTester> complex_class_tester(m, "complex_class_tester");

	complex_class_tester.def(py::init<>())
		.def("get_internal_state", &ComplexClassTester::getInternalState)
		.def("start_worker", &ComplexClassTester::startWorker)
		;
}
