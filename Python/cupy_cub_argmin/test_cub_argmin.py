# BSD 2-Clause License
#
# Copyright (c) 2025, Eijiro SHIBUSAWA
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import pytest
from typing import Dict, Any, Generator

import numpy as np
from numpy.testing import assert_allclose

@pytest.fixture(scope='function')
def setup_argmin_module() -> Generator[Dict[str, Any], Any, None]:
    if os.environ.get('NVCC') is None:
        os.environ['NVCC'] = '/usr/local/cuda/bin/nvcc'
    import cupy as cp

    max_arr = 512
    block_threads = max_arr

    dn = os.path.dirname(os.path.realpath(__file__))
    fpfn = os.path.join(dn, 'cub_argmin.cu')
    with open(fpfn, 'r') as f:
        cuda_source = f.read()

    cuda_source = cuda_source.replace('KEY_TYPE', 'int')
    cuda_source = cuda_source.replace('VALUE_TYPE', 'float')
    cuda_source = cuda_source.replace('BLOCK_THREADS', str(block_threads))

    module = cp.RawModule(code=cuda_source, backend='nvcc')
    module.compile()

    eps = 1E-7

    yield {
        'module': module,
        'max_arr': max_arr,
        'block_threads': block_threads,
        'eps':eps
    }

def test_argmin(setup_argmin_module: Generator[Dict[str, Any], Any, None]) -> None:
    import cupy as cp

    module = setup_argmin_module['module']
    max_arr = setup_argmin_module['max_arr']
    block_threads = setup_argmin_module['block_threads']
    eps = setup_argmin_module['eps']

    min_value = -1
    sz = block_threads
    for min_index in range(0, max_arr):
        assert min_index < max_arr
        arr_in = np.arange(sz, dtype=np.float32)
        arr_in[min_index] = min_value

        arr_in_gpu = cp.array(arr_in)
        out_key_gpu = cp.zeros((1,), dtype=np.int32)
        out_value_gpu = cp.zeros((1,), dtype=np.float32)
        assert arr_in_gpu.flags.c_contiguous

        gpu_func = module.get_function('BlockReduceArgminKernel')
        sz_block = block_threads,
        sz_grid = 1,
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                out_key_gpu,
                out_value_gpu,
                arr_in_gpu,
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        value_ref = min_value
        if ((min_index > 0) and (min_index < max_arr - 1)):
            # interpolation
            x_plus = arr_in[min_index + 1]
            x_minus = arr_in[min_index - 1]
            value_ref = -(x_plus - x_minus) / (2 * x_minus - 4 * min_value + 2 * x_plus);

        assert out_key_gpu.get() == min_index
        assert_allclose(value_ref, out_value_gpu.get(), rtol=eps)
