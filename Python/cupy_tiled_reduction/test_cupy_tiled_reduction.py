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
import cupy as cp

@pytest.fixture(scope='function')
def setup_module() -> Generator[Dict[str, Any], Any, None]:
    eps = 1E-6
    tile_size = 128
    tile_num = 8
    block_size = tile_num * tile_size

    dn = os.path.dirname(os.path.realpath(__file__))
    fpfn = os.path.join(dn, 'tiled_reduction.cu')
    with open(fpfn, 'r') as f:
        cuda_source = f.read()

    cuda_source = cuda_source.replace('TILE_SIZE', str(tile_size))
    cuda_source = cuda_source.replace('BLOCK_SIZE', str(block_size))

    module = cp.RawModule(code=cuda_source, enable_cooperative_groups=True)
    module.compile()

    yield {
        'module':module,
        'eps': eps,
        'tile_size': tile_size,
        'tile_num': tile_num
    }

def test_tiled_reduction(setup_module: Generator[Dict[str, Any], Any, None]) -> None:
    module = setup_module['module']
    eps = setup_module['eps']
    tile_size = setup_module['tile_size']
    tile_num = setup_module['tile_num']

    block_size = tile_num * tile_size
    length = block_size
    array_in = np.random.rand(length).astype(np.float32)
    array_in = np.reshape(array_in, (tile_num, tile_size))

    array_in_gpu = cp.array(array_in)
    array_out_gpu = cp.empty_like(array_in_gpu)
    assert array_in_gpu.flags.c_contiguous
    assert array_out_gpu.flags.c_contiguous

    sz_block = block_size,
    sz_grid = 1,
    gpu_func = module.get_function('tiledReduction')
    gpu_func(
        block=sz_block,
        grid=sz_grid,
        args=(
            array_out_gpu,
            array_in_gpu
        )
    )
    cp.cuda.runtime.deviceSynchronize()

    norms = np.linalg.norm(array_in, axis=1)
    array_out_ref = array_in / norms[:, np.newaxis]
    assert_allclose(array_out_ref, array_out_gpu.get(), rtol=eps)
