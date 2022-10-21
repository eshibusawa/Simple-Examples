# BSD 2-Clause License
#
# Copyright (c) 2022, Eijiro SHIBUSAWA
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

from unittest import TestCase
from nose.tools import ok_

import numpy as np
import cupy as cp

class TensorCoreWMMATestCase(TestCase):
    def setUp(self):
        dn = os.path.dirname(os.path.realpath(__file__))
        fpfn = os.path.join(dn, 'cupy_tensorcore.cu')
        with open(fpfn, 'r') as f:
            cuda_source = f.read()

        cuda_source_int = cuda_source.replace('MatrixABType', 'unsigned char')
        cuda_source_int = cuda_source_int.replace('MatrixCType', 'int')
        self.module_int = cp.RawModule(code=cuda_source_int)
        self.module_int.compile()

        cuda_source_float = cuda_source.replace('MatrixABType', 'half')
        cuda_source_float = cuda_source_float.replace('MatrixCType', 'float')

        self.module_float = cp.RawModule(code=cuda_source_float)
        self.module_float.compile()

    def tearDown(self):
        pass

    @staticmethod
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
        print('Elapsed Time: {} [msec]'.format(msec))

        return C.get()

    def mm_int_test(self):
        A = (4 * np.random.rand(16, 16)).astype(np.uint8)
        B = (4 * np.random.rand(16, 16)).astype(np.uint8)
        C = (64 * np.random.rand(16, 16)).astype(np.int32)

        A_gpu = cp.array(A, dtype=cp.uint8)
        B_gpu = cp.array(B, dtype=cp.uint8)
        C_gpu = cp.array(C, dtype=cp.int32)

        D = self.call_kernel(self.module_int, A_gpu, B_gpu, C_gpu)
        D_ref = np.dot(A, B) + C
        err = np.abs(D_ref - D)
        ok_(np.max(err) == 0)

    def mm_float_test(self):
        A = (4 * np.random.rand(16, 16)).astype(np.half)
        B = (4 * np.random.rand(16, 16)).astype(np.half)
        C = (64 * np.random.rand(16, 16)).astype(np.float32)

        A_gpu = cp.array(A, dtype=cp.half)
        B_gpu = cp.array(B, dtype=cp.half)
        C_gpu = cp.array(C, dtype=cp.float32)

        D = self.call_kernel(self.module_float, A_gpu, B_gpu, C_gpu)
        D_ref = np.dot(A, B) + C
        err = np.abs(D_ref - D)
        ok_(np.max(err) <= 1/32)
