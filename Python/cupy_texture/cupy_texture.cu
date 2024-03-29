//  BSD 2-Clause License
//
//  Copyright (c) 2021, Eijiro SHIBUSAWA
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

extern "C" __global__ void copyTexture(
	TEXUTURE_TEST_PIXEL_TYPE* output,
	cudaTextureObject_t tex,
	int width,
	int height)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;

	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int indexOutput = indexX + indexY * width;
	output[indexOutput] =  tex2D<TEXUTURE_TEST_PIXEL_TYPE>(tex, indexX, indexY);
}

extern "C" __global__ void copyTextureMasked(
	TEXUTURE_TEST_PIXEL_TYPE* output,
	cudaTextureObject_t tex,
	cudaTextureObject_t texMask,
	int length,
	int width,
	int height)
{
	const int indexL = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexL >= length)
	{
		return;
	}

	const uint2 index = tex2D<uint2>(texMask, indexL, 0);
	const int indexOutput = index.x + index.y * width;
	output[indexOutput] =  tex2D<TEXUTURE_TEST_PIXEL_TYPE>(tex, index.x, index.y);
}
