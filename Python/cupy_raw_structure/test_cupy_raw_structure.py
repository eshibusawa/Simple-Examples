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
from numpy.testing import assert_array_equal, assert_allclose
import cupy as cp

@pytest.fixture(scope='function')
def setup_raw_structure_module() -> Generator[Dict[str, Any], Any, None]:
    eps = 1E-5

    dn = os.path.dirname(os.path.realpath(__file__))
    fpfn = os.path.join(dn, 'cupy_raw_structure.cu')
    with open(fpfn, 'r') as f:
        cuda_source = f.read()
    module = cp.RawModule(code=cuda_source)
    module.compile()

    yield {
        'module':module,
        'eps': eps
    }

def test_raw_structure(setup_raw_structure_module: Generator[Dict[str, Any], Any, None]) -> None:
    module = setup_raw_structure_module['module']
    eps = setup_raw_structure_module['eps']

    sz_gpu = cp.empty(1, dtype=cp.int32)
    gpu_func = module.get_function("getParameters")
    gpu_func(
        block=(1,),
        grid=(1,),
        args=(
            sz_gpu,
        )
    )

    sz = sz_gpu[0].get()
    length = 128
    structure_array = cp.empty((length, sz), dtype=cp.uint8)
    gpu_func = module.get_function("fillAOS")
    gpu_func(
        block=(length,),
        grid=(1,),
        args=(
            structure_array,
            length
        )
    )

    x_array = cp.empty((length,), dtype=cp.int32)
    y_array = cp.empty((length,), dtype=cp.int32)
    value_array = cp.empty((length,), dtype=cp.float32)
    gpu_func = module.get_function("fillSOA")
    gpu_func(
        block=(length,),
        grid=(1,),
        args=(
            x_array,
            y_array,
            value_array,
            structure_array,
            length
        )
    )

    indices = np.arange(0, length, dtype=np.int32)
    assert_array_equal(indices, x_array.get())
    assert_array_equal(2 * indices, y_array.get())
    assert_allclose(np.array([1.2345], dtype=np.float32) * indices, value_array.get(), rtol=eps)
