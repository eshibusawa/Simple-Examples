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

extern "C" __global__ void sad_simd_kernel(const unsigned char* __restrict__ arrA, const unsigned char* __restrict__ arrB, short* output, int width, int height)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexY * width + indexX;
	const unsigned int *pA = reinterpret_cast<const unsigned int *>(arrA + index * (ARRAY_CHANNELS));
	const unsigned int *pB = reinterpret_cast<const unsigned int *>(arrB + index * (ARRAY_CHANNELS));

	unsigned int sum = 0;
	#pragma unroll
	for (int k = 0; k < ((ARRAY_CHANNELS)/4); k++)
	{
		sum +=  __vsadu4(pA[k], pB[k]);
	}

	output[index] = static_cast<short>(sum);
}

extern "C" __global__ void sad_kernel(const unsigned char* __restrict__ arrA, const unsigned char* __restrict__ arrB, short* output, int width, int height)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexY * width + indexX;
	const unsigned char *pA = arrA + index * (ARRAY_CHANNELS);
	const unsigned char *pB = arrB + index * (ARRAY_CHANNELS);

	short sum = 0;
	#pragma unroll
	for (int k = 0; k < (ARRAY_CHANNELS); k++)
	{
		sum += abs(static_cast<short>(pA[k]) - pB[k]);
	}
	output[index] = sum;
}
