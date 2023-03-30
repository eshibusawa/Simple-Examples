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

#ifndef COMPLEX_CLASS_HPP_
#define COMPLEX_CLASS_HPP_

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

class ComplexClassTester;

class ComplexClass
{
friend ::ComplexClassTester;
public:

public:
	ComplexClass()
		: m_internalState(-1),
		m_waitingValue(-1),
		m_stop(true)
	{
	}

	void waitEvent(int value, std::unique_lock<std::mutex> &shouldBeLocked)
	{
		m_cv.wait(shouldBeLocked, [this, value] {
			return (m_waitingValue == value) || m_stop;
		});
		shouldBeLocked.unlock();
	}

	void setEvent(int value)
	{
		m_waitingValue = value;
		m_cv.notify_all();
	}

	bool complexMethod()
	{
		std::cerr << "start method!" << std::endl;
		m_waitingValue = -1;
		const int numSteps = 5;
		for (int k = 0; k < numSteps; k++)
		{
			{
				std::unique_lock<std::mutex> lk(m_mutex);
				std::cerr << "step " << k << "..." << std::endl;
				m_internalState = k;
				waitEvent(k, lk);
			}
		}

		std::cerr << "done method!" << std::endl;
		return true;
	}

private:
	int m_internalState;
	std::condition_variable m_cv;
	std::mutex m_mutex;
	std::atomic<int> m_waitingValue;
	std::atomic<bool> m_stop;
};

#endif // COMPLEX_CLASS_HPP_
