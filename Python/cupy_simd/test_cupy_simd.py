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

import math
import os

from unittest import TestCase
from nose.tools import ok_

import numpy as np
import cupy as cp

class SIMDSADTestCase(TestCase):
    def setUp(self):
        self.channels = 16
        dn = os.path.dirname(os.path.realpath(__file__))
        fpfn = os.path.join(dn, 'cupy_simd.cu')
        with open(fpfn, 'r') as f:
            cuda_source = f.read()
        cuda_source = cuda_source.replace('ARRAY_CHANNELS', str(self.channels))

        self.gpu_module = cp.RawModule(code=cuda_source)
        self.gpu_module.compile()

    def tearDown(self):
        pass

    def sad_simd_test(self):
        w, h = 1024, 1024
        c = self.channels

        A = (255 * np.random.rand(h, w, c)).astype(np.uint8)
        B = (255 * np.random.rand(h, w, c)).astype(np.uint8)
        C_ref = np.sum(np.abs(A.astype(np.int16) - B), dtype=np.int16, axis=2)

        A_gpu = cp.array(A, dtype=cp.uint8)
        B_gpu = cp.array(B, dtype=cp.uint8)
        C_gpu = cp.empty((A_gpu.shape[0], A_gpu.shape[1]), dtype=cp.int16)
        gpu_func = self.gpu_module.get_function('sad_kernel')
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
        print('Elapsed Time: {} [msec]'.format(msec))
        err = np.abs(C_ref - C_gpu.get())
        ok_(np.max(err) == 0)

        C_gpu[:] = 0
        gpu_func = self.gpu_module.get_function('sad_simd_kernel')
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
        print('Elapsed Time: {} [msec]'.format(msec))
        err = np.abs(C_ref - C_gpu.get())
        ok_(np.max(err) == 0)
