# BSD 2-Clause License
#
# Copyright (c) 2023, Eijiro SHIBUSAWA
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

from unittest import TestCase
from nose.tools import ok_

import numpy as np
import cupy as cp

import util_cuda_array as uca
import device_memory_container

class UtilCudaArrayTestCase(TestCase):
    def setUp(self):
        self.eps = 1E-7

    def tearDown(self):
        pass

    def to_array_test(self):
        dmc = device_memory_container.device_memory_container()
        arr_ref = np.array(dmc.get(), dtype=np.float32)
        arr = np.arange(0, arr_ref.shape[0], dtype=arr_ref.dtype)
        err = np.abs(arr_ref - arr)
        ok_(np.max(err) < self.eps)

        d = dmc.get_ptr()
        ptr = d["ptr"]
        arr_gpu = uca.get_array_from_ptr(self, ptr, arr_ref.shape, arr_ref.dtype)
        arr_gpu_ref = cp.arange(0, arr_ref.shape[0], dtype=arr_ref.dtype)
        err = cp.abs(arr_gpu_ref - arr_gpu)
        ok_(cp.max(err).get() < self.eps)

    def from_array_test(self):
        dmc = device_memory_container.device_memory_container()
        arr = np.array(dmc.get(), dtype=np.float32)

        d = dmc.get_ptr()
        ptr = d["ptr"]
        arr_gpu_ref = cp.random.rand(arr.shape[0]).astype(cp.float32)
        arr_gpu = uca.get_array_from_ptr(self, ptr, arr_gpu_ref.shape, arr_gpu_ref.dtype)
        arr_gpu[:] = arr_gpu_ref
        err = cp.abs(arr_gpu_ref - arr_gpu)
        ok_(cp.max(err).get() < self.eps)
