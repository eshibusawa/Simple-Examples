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
from numpy.testing import assert_equal, assert_allclose
import cupy as cp

@pytest.fixture(scope='module')
def setup_module() -> Generator[Dict[str, Any], Any, None]:
    dn = os.path.dirname(os.path.realpath(__file__))
    fpfn = os.path.join(dn, 'cupy_tensorcore.cu')
    with open(fpfn, 'r') as f:
        cuda_source = f.read()

    cuda_source_int = cuda_source.replace('MatrixABType', 'unsigned char')
    cuda_source_int = cuda_source_int.replace('MatrixCType', 'int')
    module_int = cp.RawModule(code=cuda_source_int)
    module_int.compile()

    cuda_source_float = cuda_source.replace('MatrixABType', 'half')
    cuda_source_float = cuda_source_float.replace('MatrixCType', 'float')

    module_float = cp.RawModule(code=cuda_source_float)
    module_float.compile()

    yield {
        'module_int': module_int,
        'module_float': module_float
    }

def call_kernel(module, A, B, C):
    gpu_func = module.get_function('wmma_16x16')
    block = 32, 1
    grid = 1, 1

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    gpu_func(
        block=block,
        grid=grid,
        args=(
            A,
            B,
            C,
        )
    )
    cp.cuda.runtime.deviceSynchronize()
    end.record()
    end.synchronize()

    msec = cp.cuda.get_elapsed_time(start, end)
    print(f'Elapsed Time: {msec} [msec]')

    return C.get()

def test_mm_int(setup_module: Generator[Dict[str, Any], Any, None]) -> None:
    module = setup_module['module_int']
    A = (4 * np.random.rand(16, 16)).astype(np.uint8)
    B = (4 * np.random.rand(16, 16)).astype(np.uint8)
    C = (64 * np.random.rand(16, 16)).astype(np.int32)

    A_gpu = cp.array(A, dtype=cp.uint8)
    B_gpu = cp.array(B, dtype=cp.uint8)
    C_gpu = cp.array(C, dtype=cp.int32)

    D = call_kernel(module, A_gpu, B_gpu, C_gpu)
    D_ref = np.dot(A, B) + C
    assert_equal(D_ref, D)

def test_mm_float(setup_module: Generator[Dict[str, Any], Any, None]) -> None:
    module = setup_module['module_float']
    A = (4 * np.random.rand(16, 16)).astype(np.half)
    B = (4 * np.random.rand(16, 16)).astype(np.half)
    C = (64 * np.random.rand(16, 16)).astype(np.float32)

    A_gpu = cp.array(A, dtype=cp.half)
    B_gpu = cp.array(B, dtype=cp.half)
    C_gpu = cp.array(C, dtype=cp.float32)

    D = call_kernel(module, A_gpu, B_gpu, C_gpu)
    D_ref = np.dot(A, B) + C
    assert_allclose(D_ref, D, rtol=1/32, atol=0)
