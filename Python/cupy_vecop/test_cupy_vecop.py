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

import os

from unittest import TestCase
from nose.tools import ok_

import numpy as np
import cupy as cp

class VecOpTestCase(TestCase):
    def setUp(self):
        self.eps = 1E-5

        dn = os.path.dirname(os.path.realpath(__file__))
        fpfn = os.path.join(dn, 'cupy_vecop.cu')
        # load raw kernel
        with open(fpfn, 'r') as f:
            cuda_source = f.read()
        self.module = cp.RawModule(code=cuda_source)
        self.module.compile()

    def tearDown(self):
        pass

    def plus_test(self):
        sz = 1024
        A = np.random.rand(sz).astype(np.float32)
        B = np.random.rand(sz).astype(np.float32)
        A_gpu = cp.array(A, dtype=cp.float32)
        B_gpu = cp.array(B, dtype=cp.float32)
        C_gpu = cp.empty_like(A)

        gpu_func = self.module.get_function("vecPlus")
        gpu_func(
            block=(sz,),
            grid=(1,),
            args=(
                C_gpu,
                A_gpu,
                B_gpu,
                sz
            )
        )
        cp.cuda.runtime.deviceSynchronize()

        C = A + B
        err = np.abs(C - C_gpu.get())
        ok_(np.max(err) < self.eps)

    def minus_test(self):
        sz = 1024
        A = np.random.rand(sz).astype(np.float32)
        B = np.random.rand(sz).astype(np.float32)
        A_gpu = cp.array(A, dtype=cp.float32)
        B_gpu = cp.array(B, dtype=cp.float32)
        C_gpu = cp.empty_like(A)

        gpu_func = self.module.get_function("vecMinus")
        gpu_func(
            block=(sz,),
            grid=(1,),
            args=(
                C_gpu,
                A_gpu,
                B_gpu,
                sz
            )
        )
        cp.cuda.runtime.deviceSynchronize()

        C = A - B
        err = np.abs(C - C_gpu.get())
        ok_(np.max(err) < self.eps)
