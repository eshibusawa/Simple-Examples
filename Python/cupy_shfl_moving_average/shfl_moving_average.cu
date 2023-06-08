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

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ inline int clamp_(int index, int length)
{
	return min(max(index, 0), length - 1);
}

extern "C" __global__ void movingAverageShfl(
	float *output,
	const float* __restrict__ input,
	int length)
{
	static const int HALF_WINDOW = (WINDOW_SIZE)/2;
	static const int TILE_SIZE = 32;
	auto cta = cg::this_thread_block();
	int offsetGlobal = cta.group_index().x * BLOCK_SIZE;
	const int cta_tid = cta.thread_rank();
	// work on tile
	auto tile = cg::tiled_partition<TILE_SIZE>(cta);
	const int tile_tid= tile.thread_rank();
	#pragma unroll
	for (int offsetTile = 0; offsetTile < WINDOW_SIZE; offsetTile++)
	{
		float sum = input[clamp_(offsetGlobal + cta_tid + offsetTile - HALF_WINDOW, length)] / WINDOW_SIZE;
		#pragma unroll
		for (int k = HALF_WINDOW; k > 0; k /= 2)
		{
			sum += tile.shfl_down(sum, k);
		}

		if (tile_tid % WINDOW_SIZE == 0)
		{
			output[offsetGlobal + cta_tid + offsetTile] = sum;
		}
	}
}

extern "C" __global__ void movingAverageNaive(
	float *output,
	const float* __restrict__ input,
	int length)
{
	static const int HALF_WINDOW = (WINDOW_SIZE)/2;

	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= length)
	{
		return;
	}

	float sum = 0.f;
	int indexClamped;
	for (int k = 0; k < (WINDOW_SIZE); k++)
	{
		indexClamped = clamp_(index + k - HALF_WINDOW, length);
		sum += input[indexClamped];
	}
	output[index] = sum / (WINDOW_SIZE);
}
