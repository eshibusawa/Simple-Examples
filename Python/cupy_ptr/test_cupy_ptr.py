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

import pytest
from typing import Dict, Any, Generator

import numpy as np
from numpy.testing import assert_allclose
import cupy as cp

import util_cuda_array as uca
import device_memory_container

@pytest.fixture(scope='function')
def setup_ptr_module() -> Generator[Dict[str, Any], Any, None]:
    eps = 1E-7
    yield {
        'eps': eps,
    }

def test_to_array(setup_ptr_module: Generator[Dict[str, Any], Any, None]) -> None:
    eps = setup_ptr_module['eps']

    dmc = device_memory_container.device_memory_container()
    arr_ref = np.array(dmc.get(), dtype=np.float32)
    arr = np.arange(0, arr_ref.shape[0], dtype=arr_ref.dtype)
    assert_allclose(arr_ref, arr, rtol=eps)

    d = dmc.get_ptr()
    ptr = d['ptr']
    assert arr_ref.dtype == cp.float32
    arr_gpu = uca.get_array_from_ptr(dmc, ptr, arr_ref.shape, arr_ref.dtype)
    arr_gpu_ref = cp.arange(0, arr_ref.shape[0], dtype=arr_ref.dtype)
    assert_allclose(arr_gpu_ref.get(), arr_gpu.get(), rtol=eps)

def test_from_array(setup_ptr_module: Generator[Dict[str, Any], Any, None]) -> None:
    eps = setup_ptr_module['eps']

    dmc = device_memory_container.device_memory_container()
    arr = np.array(dmc.get(), dtype=np.float32)

    d = dmc.get_ptr()
    ptr = d['ptr']
    arr_gpu_ref = cp.random.rand(arr.shape[0]).astype(cp.float32)
    assert arr_gpu_ref.dtype == cp.float32
    arr_gpu = uca.get_array_from_ptr(dmc, ptr, arr_gpu_ref.shape, arr_gpu_ref.dtype)
    arr_gpu[:] = arr_gpu_ref
    assert_allclose(arr_gpu_ref.get(), arr_gpu.get(), rtol=eps)
