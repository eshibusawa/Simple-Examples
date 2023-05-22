//  BSD 2-Clause License
//
//  Copyright (c) 2023, Eijiro SHIBUSAWA
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

template<typename Float>
struct PlusOp
{
	__device__ static Float op(Float a, Float b)
	{
		return a + b;
	}
};

template<typename Float>
struct MinusOp
{
	__device__ static Float op(Float a, Float b)
	{
		return a - b;
	}
};

template<typename Op, typename Float>
__device__ void vecOp(Float *C, const Float * __restrict__ A, const Float * __restrict__ B, int sz)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sz)
	{
		return;
	}
	C[index] = Op::op(A[index], B[index]);
}

extern "C" __global__ void vecPlus(float *C, const float * __restrict__ A, const float * __restrict__ B, int sz)
{
	vecOp<PlusOp<float>, float>(C, A, B, sz);
}

extern "C" __global__ void vecMinus(float *C, const float * __restrict__ A, const float * __restrict__ B, int sz)
{
	vecOp<MinusOp<float>, float>(C, A, B, sz);
}
