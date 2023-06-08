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

class RadixSortTestCase(TestCase):
    def setUp(self):
        if os.environ.get('NVCC') is None:
            os.environ['NVCC'] = '/usr/local/cuda/bin/nvcc'
        import cupy as cp

        self.max_arr = 1024
        self.block_threads = 128
        self.item_per_threads = 13

        dn = os.path.dirname(os.path.realpath(__file__))
        fpfn = os.path.join(dn, 'cub_radix_sort.cu')
        with open(fpfn, 'r') as f:
            cuda_source = f.read()

        cuda_source = cuda_source.replace('KEY_TYPE', 'int')
        cuda_source = cuda_source.replace('BLOCK_THREADS', str(self.block_threads))
        cuda_source = cuda_source.replace('ITEMS_PER_THREAD', str(self.item_per_threads))

        self.module = cp.RawModule(code=cuda_source, backend='nvcc')
        self.module.compile()

    def tearDown(self):
        pass

    def radix_sort_test(self):
        import cupy as cp

        sz = self.block_threads * self.item_per_threads
        arr_in = (self.max_arr * np.random.rand(sz)).astype(np.int32)
        arr_in_gpu = cp.array(arr_in)
        arr_out_gpu = cp.zeros_like(arr_in_gpu)

        gpu_func = self.module.get_function('BlockSortKernel')
        sz_block = self.block_threads,
        sz_grid = 1,
        gpu_func(
            block=sz_block,
            grid=sz_grid,
            args=(
                arr_in_gpu,
                arr_out_gpu
            )
        )
        cp.cuda.runtime.deviceSynchronize()

        arr_out_ref = np.sort(arr_in)
        arr_out = arr_out_gpu.get()
        err = np.abs(arr_out_ref- arr_out)
        ok_(np.max(err) == 0)
