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

class ShflMovingAverageTestCase(TestCase):
    def setUp(self):
        if os.environ.get('NVCC') is None:
            os.environ['NVCC'] = '/usr/local/cuda/bin/nvcc'
        import cupy as cp

        self.eps = 1E-5
        self.window_size = 8
        self.block_size = 128

        dn = os.path.dirname(os.path.realpath(__file__))
        fpfn = os.path.join(dn, 'shfl_moving_average.cu')
        # load raw kernel
        with open(fpfn, 'r') as f:
            cuda_source = f.read()

        cuda_source = cuda_source.replace('WINDOW_SIZE', str(self.window_size))
        cuda_source = cuda_source.replace('BLOCK_SIZE', str(self.block_size))

        self.module = cp.RawModule(code=cuda_source, backend='nvcc')
        self.module.compile()

    def tearDown(self):
        pass

    def moving_average_test(self):
        import cupy as cp
        length = 1 << 14
        array_in = np.random.rand(length).astype(np.float32)

        array_in_gpu = cp.array(array_in)
        array_out_ref_gpu = cp.empty_like(array_in_gpu)
        assert array_in_gpu.flags.c_contiguous
        assert array_out_ref_gpu.flags.c_contiguous

        sz_block = 1024,
        sz_grid = math.ceil(array_in_gpu.shape[0] / sz_block[0]),
        gpu_func = self.module.get_function("movingAverageNaive")
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        start.synchronize()
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                array_out_ref_gpu,
                array_in_gpu,
                array_in_gpu.shape[0]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        end.record()
        end.synchronize()
        msec_naive = cp.cuda.get_elapsed_time(start, end)

        array_out_gpu = cp.full_like(array_in_gpu, -1)
        assert array_out_gpu.flags.c_contiguous
        sz_block = self.block_size,
        sz_grid = math.ceil(array_in_gpu.shape[0] / sz_block[0]),
        gpu_func = self.module.get_function("movingAverageShfl")
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        start.synchronize()
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                array_out_gpu,
                array_in_gpu,
                array_in_gpu.shape[0]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        end.record()
        end.synchronize()
        msec_shfl = cp.cuda.get_elapsed_time(start, end)

        array_out_ref = array_out_ref_gpu.get()
        array_out = array_out_gpu.get()
        err = np.abs(array_out_ref - array_out)
        ok_(np.max(err) < self.eps)

        print('')
        print('Elapsed Time Naive: {} [msec]'.format(msec_naive))
        print('Elapsed Time Shfl: {} [msec]'.format(msec_shfl))
