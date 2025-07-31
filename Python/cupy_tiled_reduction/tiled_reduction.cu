//  BSD 2-Clause License
//
//  Copyright (c) 2025, Eijiro SHIBUSAWA
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

#if __CUDACC_VER_MAJOR__ < 12
	#error "CUDA Toolkit 12 or newer is required."
#endif
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

extern "C" __global__ void tiledReduction(float* output,
		const float* __restrict__ input)
{
	// see "8.2.3. CUDA 12.0" of CUDA C++ Programming Guide v12.6
#if __CUDA_ARCH__ >= 800
	cg::thread_block cta = cg::this_thread_block();
#else
	__shared__ cg::block_tile_memory<BLOCK_SIZE> shared;
	cg::thread_block cta = cg::this_thread_block(shared);
#endif
	const int index = cta.thread_rank();
	if (index >= BLOCK_SIZE)
	{
		return;
	}
	float threadData = input[index];
	auto tile = cg::tiled_partition<TILE_SIZE>(cta);
	float sqrData = threadData * threadData;
	tile.sync();

	float sqrSum = cg::reduce(tile, sqrData, cg::plus<float>());
	tile.sync();
	output[index] = threadData / sqrtf(sqrSum);
}
