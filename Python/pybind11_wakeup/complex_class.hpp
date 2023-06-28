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

#include "mt_debugger.hpp"

#include <iostream>

class ComplexClass
{
public:
	ComplexClass()
	{
	}

	bool complexMethod(int numSteps)
	{
		std::cerr << "start method!" << std::endl;
		int internalState = 0;
		m_debugger.push(&internalState);
		for (int k = 0; k < numSteps; k++)
		{
			m_debugger.conditionalStep(true);
			std::cerr << "step " << k << "..." << std::endl;
			std::cerr << " state " << internalState << std::endl;
		}

		std::cerr << "done method!" << std::endl;
		return true;
	}

private:
	MTDebugger m_debugger;
};

#endif // COMPLEX_CLASS_HPP_
