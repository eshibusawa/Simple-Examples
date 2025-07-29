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
from numpy.testing import assert_allclose
import cupy as cp
import torch

@pytest.fixture(scope='function')
def setup_module() -> Generator[Dict[str, Any], Any, None]:
    eps = 1E-6
    sz = 1080, 1920
    scale_length = 1

    dn = os.path.dirname(os.path.realpath(__file__))
    fpfn = os.path.join(dn, 'device_code.cu')
    with open(fpfn, 'r') as f:
        cuda_source = f.read()
    cuda_source = cuda_source.replace("SCALE_LENGTH", str(scale_length))
    cuda_source = cuda_source.replace("HEIGHT", str(sz[0]))
    cuda_source = cuda_source.replace("WIDTH", str(sz[1]))
    module = cp.RawModule(code=cuda_source)

    yield {
        'module':module,
        'eps': eps,
        'sz': sz,
        'scale_length': scale_length
    }

def test_scale_grid(setup_module: Generator[Dict[str, Any], Any, None]) -> None:
    module = setup_module['module']
    eps = setup_module['eps']
    sz = setup_module['sz']
    scale_length = setup_module['scale_length']

    xy = np.empty((sz[0], sz[1], 2), dtype=np.float32)
    xy[:,:,0] = np.arange(0, sz[1])[np.newaxis,:]
    xy[:,:,1] = np.arange(0, sz[0])[:,np.newaxis]

    scale = np.random.rand(scale_length).astype(np.float32)[0]
    xy_scale_ref = scale * xy

    # upload scale value to GPU constant memory
    scale_ptr = module.get_global('g_scale')
    scale_gpu = cp.ndarray((scale_length,), cp.float32, scale_ptr)
    scale_gpu[...] = scale

    xy_scale_gpu = torch.empty((sz[0], sz[1], 2), dtype=torch.float32, device='cuda')
    sz_block = 16, 16
    sz_grid = math.ceil(sz[1] / sz_block[1]), math.ceil(sz[0] / sz_block[0])
    gpu_func = module.get_function('scaleGrid')
    gpu_func(
        block=sz_block,
        grid=sz_grid,
        args=(
            xy_scale_gpu.data_ptr(),
        )
    )
    xy_scale = xy_scale_gpu.cpu().numpy()
    assert_allclose(xy_scale_ref, xy_scale, rtol=eps, atol=0)
