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

import util_array as ua
import vector_container

@pytest.fixture(scope='module')
def setup() -> Generator[Dict[str, Any], Any, None]:
    eps = 1E-7

    yield {
        'eps': eps,
    }

def test_from_array(setup: Generator[Dict[str, Any], Any, None]) -> None:
    eps = setup['eps']

    vc = vector_container.vector_container()
    arr_ref = np.array(vc.get(), dtype=np.float32)
    arr = np.arange(0, arr_ref.shape[0], dtype=arr_ref.dtype)
    assert_allclose(arr_ref, arr, rtol=eps, atol=0)

    d = vc.get_ptr()
    ptr = d["ptr"]
    arr_rand_ref = np.random.rand(arr_ref.shape[0]).astype(arr_ref.dtype)
    ua.copy_from_array(arr_rand_ref, ptr)
    arr_rand = np.array(vc.get(), dtype=arr_ref.dtype)

    assert_allclose(arr_rand_ref, arr_rand, rtol=eps, atol=0)


def test_to_array(setup: Generator[Dict[str, Any], Any, None]) -> None:
    eps = setup['eps']

    vc = vector_container.vector_container()
    arr_ref = np.array(vc.get(), dtype=np.float32)

    d = vc.get_ptr()
    ptr = d["ptr"]
    arr = np.random.rand(arr_ref.shape[0]).astype(arr_ref.dtype)
    ua.copy_to_array(arr, ptr)
    assert_allclose(arr_ref, arr, rtol=eps, atol=0)
