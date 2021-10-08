// BSD 2-Clause License
//
// Copyright (c) 2021, Eijiro SHIBUSAWA
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

#include <torch/torch.h>

#include <cmath>

__constant__ float g_scale[1];

namespace {
inline int iDivUp(int a, int b)
{
	return static_cast<int>(std::ceil(static_cast<float>(a)/b));
}

template <typename scalar_t>
__global__ void scaleGridCUDAKernel(
	scalar_t* __restrict__ output,
	int height,
	int width)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;

	if ((column >= width) || (row >= height))
	{
		return;
	}
	const int w2 = 2 * width, c2 = 2 * column;
	const int index = c2 + row * w2;
	output[index] = column * (g_scale[0]);
	output[index + 1] = row * (g_scale[0]);
}
}

torch::Tensor scaleGridCUDA(const std::vector<int> &outputSize, float scale)
{
    cudaError err = cudaMemcpyToSymbol(g_scale, &scale, sizeof(float));
    if (cudaSuccess != err)
	{
		std::cerr << "CUDA Error: " << __FILE__ << ":" << __LINE__ << std::endl;
		std::cerr << "   " << cudaGetErrorString(err) << std::endl;
    }

	const int w = outputSize[1], h = outputSize[0];
	auto OutputOptions =
	torch::TensorOptions()
		.dtype(torch::kFloat)
		.device(torch::kCUDA, 0);
	auto output = torch::empty({h, w, 2}, OutputOptions);

	const dim3 threads(32, 32);
	const dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	AT_DISPATCH_FLOATING_TYPES(output.type(), "scaleGridCUDAKernelDispatch", ([&] {
		scaleGridCUDAKernel<scalar_t><<<blocks, threads>>>(
				output.data_ptr<scalar_t>(),
				h,
				w);
	}));

	return output;
}
