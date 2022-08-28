//  BSD 2-Clause License
//
//  Copyright (c) 2022, Eijiro SHIBUSAWA
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this
//     list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

struct SimpleStruct
{
	__device__ SimpleStruct():
		x(0),
		y(0),
		value(0)
	{
	}

	int x, y;
	float value;
};

extern "C" __global__ void getParameters(
	int *sz)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > 0)
	{
		return;
	}
	*sz = sizeof(SimpleStruct);
}

extern "C" __global__ void fillAOS(
	unsigned char *pAOS,
	int length
	)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= length)
	{
		return;
	}
	SimpleStruct *p = reinterpret_cast<SimpleStruct *>(pAOS + index * sizeof(SimpleStruct));
	p->x = index;
	p->y = index * 2;
	p-> value = index * 1.2345f;
}

extern "C" __global__ void fillSOA(
	int *x,
	int *y,
	float *value,
	const unsigned char* __restrict__ pAOS,
	int length
	)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= length)
	{
		return;
	}
	const SimpleStruct *p = reinterpret_cast<const SimpleStruct *>(pAOS + index * sizeof(SimpleStruct));
	x[index] = p->x;
	y[index] = p->y;
	value[index] = p->value;
}
