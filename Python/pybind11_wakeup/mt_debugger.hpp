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

#ifndef MTDEBUGGER_HPP_
#define MTDEBUGGER_HPP_

#include <atomic>
#include <condition_variable>
#include <thread>
#include <vector>

class MTDebugger
{
public:
	int m_stateCounter;
	int m_internalState;
	int m_debugSolverIteration;
	std::condition_variable m_cv;
	std::mutex m_mutex;
	std::unique_lock<std::mutex> m_lk;
	std::atomic<int> m_previousState;
	std::atomic<int> m_waitingValue;
	std::atomic<bool> m_stop;
	std::atomic<bool> m_enable;
	std::vector<void *> m_data;

	static const int m_waitMs;

public:
	MTDebugger():
	m_stateCounter(-1),
	m_internalState(-1),
	m_debugSolverIteration(-1),
	m_lk(m_mutex, std::defer_lock),
	m_previousState(-1),
	m_waitingValue(-1),
	m_stop(true),
	m_enable(false)
	{
	}

	void waitEvent(int value)
	{
		m_cv.wait(m_lk, [this, value] {
			return (m_waitingValue == value) || m_stop;
			});
		m_lk.unlock();
	}

	void setEvent(int value)
	{
		m_waitingValue = value;
		m_cv.notify_all();
	}

	inline void conditionalStep(bool pred)
	{
		if ((!pred) || !(m_enable))
		{
			return;
		}
		// step
		m_lk.lock();
		m_stateCounter++;
		m_internalState = m_stateCounter;
		waitEvent(m_stateCounter);
	}

	inline void clear()
	{
		m_stateCounter = -1;
		m_previousState = -1;
		m_waitingValue = -1;
		m_data.clear();
	}

	void incrementInternalState()
	{
		int ret = m_previousState;
		do
		{
			{
				std::lock_guard<std::mutex> lk(m_mutex);
				ret = m_internalState;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(m_waitMs)); // loop wait, for simplicity
		} while (ret == m_previousState);
		m_previousState = ret;
	}

	void resume()
	{
		setEvent(m_previousState);
	}

	void stop()
	{
		m_stop = true;
		m_cv.notify_all();
		std::this_thread::sleep_for(std::chrono::milliseconds(m_waitMs));
	}

	void push(void *d)
	{
		std::lock_guard<std::mutex> lk(m_mutex);
		m_data.push_back(d);
	}

	void get(size_t index, std::vector<void *> &d)
	{
		const size_t t = index + 1;
		size_t ret = 0;
		while (ret < t)
		{
			std::lock_guard<std::mutex> lk(m_mutex);
			ret = m_data.size();
		}
		{
			std::lock_guard<std::mutex> lk(m_mutex);
			d.push_back(m_data[index]);
		}
	}
};

const int MTDebugger::m_waitMs(5);

#endif // MTDEBUGGER_HPP_
