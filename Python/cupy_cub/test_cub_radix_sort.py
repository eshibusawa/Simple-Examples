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
from numpy.testing import assert_array_equal

@pytest.fixture(scope='function')
def setup_radix_sort_module() -> Generator[Dict[str, Any], Any, None]:
    if os.environ.get('NVCC') is None:
        os.environ['NVCC'] = '/usr/local/cuda/bin/nvcc'
    import cupy as cp

    max_arr = 1024
    block_threads = 128
    item_per_threads = 13

    dn = os.path.dirname(os.path.realpath(__file__))
    fpfn = os.path.join(dn, 'cub_radix_sort.cu')

    with open(fpfn, 'r') as f:
        cuda_source = f.read()

    cuda_source = cuda_source.replace('KEY_TYPE', 'int')
    cuda_source = cuda_source.replace('BLOCK_THREADS', str(block_threads))
    cuda_source = cuda_source.replace('ITEMS_PER_THREAD', str(item_per_threads))

    module = cp.RawModule(code=cuda_source, backend='nvcc')
    module.compile()

    yield {
        'module': module,
        'max_arr': max_arr,
        'block_threads': block_threads,
        'item_per_threads': item_per_threads
    }

def test_radix_sort(setup_radix_sort_module: Generator[Dict[str, Any], Any, None]) -> None:
    import cupy as cp

    module = setup_radix_sort_module['module']
    max_arr = setup_radix_sort_module['max_arr']
    block_threads = setup_radix_sort_module['block_threads']
    item_per_threads = setup_radix_sort_module['item_per_threads']

    sz = block_threads * item_per_threads
    arr_in = (max_arr * np.random.rand(sz)).astype(np.int32)
    arr_in_gpu = cp.array(arr_in)
    arr_out_gpu = cp.zeros_like(arr_in_gpu)

    gpu_func = module.get_function('BlockSortKernel')
    sz_block = block_threads,
    sz_grid = 1,
    gpu_func(
        block=sz_block,
        grid=sz_grid,
        args=(
            arr_in_gpu,
            arr_out_gpu
        )
    )
    cp.cuda.runtime.deviceSynchronize()

    arr_out_ref = np.sort(arr_in)
    arr_out = arr_out_gpu.get()
    assert_array_equal(arr_out_ref, arr_out)
