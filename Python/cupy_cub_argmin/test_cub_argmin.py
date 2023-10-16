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

class ArgminTestCase(TestCase):
    def setUp(self):
        if os.environ.get('NVCC') is None:
            os.environ['NVCC'] = '/usr/local/cuda/bin/nvcc'
        import cupy as cp

        self.max_arr = 512
        self.block_threads = self.max_arr

        dn = os.path.dirname(os.path.realpath(__file__))
        fpfn = os.path.join(dn, 'cub_argmin.cu')
        with open(fpfn, 'r') as f:
            cuda_source = f.read()

        cuda_source = cuda_source.replace('KEY_TYPE', 'int')
        cuda_source = cuda_source.replace('VALUE_TYPE', 'float')
        cuda_source = cuda_source.replace('BLOCK_THREADS', str(self.block_threads))

        self.module = cp.RawModule(code=cuda_source, backend='nvcc')
        self.module.compile()

        self.eps = 1E-7

    def tearDown(self):
        pass

    def argmin_test(self):
        import cupy as cp

        min_value = -1
        sz = self.block_threads
        for min_index in range(0, self.max_arr):
            assert min_index < self.max_arr
            arr_in = np.arange(sz, dtype=np.float32)
            arr_in[min_index] = min_value

            arr_in_gpu = cp.array(arr_in)
            out_key_gpu = cp.zeros((1,), dtype=np.int32)
            out_value_gpu = cp.zeros((1,), dtype=np.float32)
            assert arr_in_gpu.flags.c_contiguous

            gpu_func = self.module.get_function('BlockReduceArgminKernel')
            sz_block = self.block_threads,
            sz_grid = 1,
            gpu_func(
                block=sz_block,
                grid=sz_grid,
                args=(
                    out_key_gpu,
                    out_value_gpu,
                    arr_in_gpu,
                )
            )
            cp.cuda.runtime.deviceSynchronize()
            value_ref = min_value
            if ((min_index > 0) and (min_index < self.max_arr - 1)):
                # interpolation
                x_plus = arr_in[min_index + 1]
                x_minus = arr_in[min_index - 1]
                value_ref = -(x_plus - x_minus) / (2 * x_minus - 4 * min_value + 2 * x_plus);

            ok_(out_key_gpu.get() == min_index)
            err = np.abs(out_value_gpu.get() - value_ref)
            ok_(err < self.eps)
