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
import math
import pytest
from typing import Dict, Any, Generator

import numpy as np
from numpy.testing import assert_equal
import cupy as cp

@pytest.fixture(scope='function')
def setup_module() -> Generator[Dict[str, Any], Any, None]:
    channels = 16
    dn = os.path.dirname(os.path.realpath(__file__))
    fpfn = os.path.join(dn, 'cupy_simd.cu')
    with open(fpfn, 'r') as f:
        cuda_source = f.read()
    cuda_source = cuda_source.replace('ARRAY_CHANNELS', str(channels))

    module = cp.RawModule(code=cuda_source)
    module.compile()

    yield {
        'module':module,
        'channels': channels
    }

def test_sad_simd(setup_module: Generator[Dict[str, Any], Any, None]) -> None:
    module = setup_module['module']
    channels = setup_module['channels']
    w, h = 1024, 1024

    A = (255 * np.random.rand(h, w, channels)).astype(np.uint8)
    B = (255 * np.random.rand(h, w, channels)).astype(np.uint8)
    C_ref = np.sum(np.abs(A.astype(np.int16) - B), dtype=np.int16, axis=2)

    A_gpu = cp.array(A, dtype=cp.uint8)
    B_gpu = cp.array(B, dtype=cp.uint8)
    C_gpu = cp.empty((A_gpu.shape[0], A_gpu.shape[1]), dtype=cp.int16)
    gpu_func = module.get_function('sad_kernel')
    block = 32, 32
    grid = math.ceil(A_gpu.shape[1] / block[0]), math.ceil(A_gpu.shape[0] / block[1])
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    gpu_func(
        block=block,
        grid=grid,
        args=(
            A_gpu,
            B_gpu,
            C_gpu,
            A_gpu.shape[1],
            A_gpu.shape[0]
        )
    )
    cp.cuda.runtime.deviceSynchronize()
    end.record()
    end.synchronize()
    msec = cp.cuda.get_elapsed_time(start, end)
    print('')
    print(f'Elapsed Time: {msec} [msec]')
    assert_equal(C_ref, C_gpu.get())

    C_gpu[:] = 0
    gpu_func = module.get_function('sad_simd_kernel')
    block = 32, 32
    grid = math.ceil(A_gpu.shape[1] / block[0]), math.ceil(A_gpu.shape[0] / block[1])
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    gpu_func(
        block=block,
        grid=grid,
        args=(
            A_gpu,
            B_gpu,
            C_gpu,
            A_gpu.shape[1],
            A_gpu.shape[0]
        )
    )
    cp.cuda.runtime.deviceSynchronize()
    end.record()
    end.synchronize()
    msec = cp.cuda.get_elapsed_time(start, end)
    print(f'Elapsed Time: {msec} [msec]')
    assert_equal(C_ref, C_gpu.get())
