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

from simple_pybind import simple_class

@pytest.fixture(scope='module')
def setup() -> Generator[Dict[str, Any], Any, None]:
    eps = 1E-5
    arr = np.random.rand(3, 3).astype(np.float32)

    yield {
        'eps': eps,
        'arr': arr
    }

class simple_class_settings:
    def __init__(self):
        self.value = 1
        self.computation_mode = simple_class.computation_mode.add

def test_add_operator(setup: Generator[Dict[str, Any], Any, None]) -> None:
    eps = setup['eps']
    arr = setup['arr']

    scs = simple_class_settings()
    scs.value = np.random.rand(1).astype(np.float32)[0]
    scs.computation_mode = simple_class.computation_mode.add

    sc = simple_class()
    arr2 = sc.compute(scs, arr)

    arr2_ref = arr + scs.value
    assert_allclose(arr2_ref, arr2, rtol=eps, atol=0)

def test_sub_operator(setup: Generator[Dict[str, Any], Any, None]) -> None:
    eps = setup['eps']
    arr = setup['arr']

    scs = simple_class_settings()
    scs.value = np.random.rand(1).astype(np.float32)[0]
    scs.computation_mode = simple_class.computation_mode.sub

    sc = simple_class()
    arr2 = sc.compute(scs, arr)

    arr2_ref = arr - scs.value
    assert_allclose(arr2_ref, arr2, rtol=eps, atol=0)

def test_mul_operator(setup: Generator[Dict[str, Any], Any, None]) -> None:
    eps = setup['eps']
    arr = setup['arr']

    scs = simple_class_settings()
    scs.value = np.random.rand(1).astype(np.float32)[0]
    scs.computation_mode = simple_class.computation_mode.mul

    sc = simple_class()
    arr2 = sc.compute(scs, arr)

    arr2_ref = arr * scs.value
    assert_allclose(arr2_ref, arr2, rtol=eps, atol=0)

def test_div_operator(setup: Generator[Dict[str, Any], Any, None]) -> None:
    eps = setup['eps']
    arr = setup['arr']

    scs = simple_class_settings()
    scs.value = np.random.rand(1).astype(np.float32)[0]
    scs.computation_mode = simple_class.computation_mode.div

    sc = simple_class()
    arr2 = sc.compute(scs, arr)

    arr2_ref = arr / scs.value
    assert_allclose(arr2_ref, arr2, rtol=eps, atol=0)
