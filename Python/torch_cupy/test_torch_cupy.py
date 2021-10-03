# BSD 2-Clause License
#
# Copyright (c) 2021, Eijiro SHIBUSAWA
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
import math

from unittest import TestCase
from nose.tools import ok_

import numpy as np
import cupy as cp
import torch

class TorchCupyTestCase(TestCase):
    def setUp(self):
        self.eps = 1E-6
        self.sz = 1080, 1920
        self.scale_length = 1

        dn = os.path.dirname(os.path.realpath(__file__))
        fpfn = os.path.join(dn, 'device_code.cu')
        # load raw kernel
        with open(fpfn, 'r') as f:
            cuda_source = f.read()
        cuda_source = cuda_source.replace("SCALE_LENGTH", str(self.scale_length))
        cuda_source = cuda_source.replace("HEIGHT", str(self.sz[0]))
        cuda_source = cuda_source.replace("WIDTH", str(self.sz[1]))
        self.module = cp.RawModule(code=cuda_source)
        self.scale_grid_gpu = self.module.get_function("scaleGrid")

    def tearDown(self):
        pass

    def scale_grid_test(self):
        xy = np.empty((self.sz[0], self.sz[1], 2), dtype=np.float32)
        xy[:,:,0] = np.arange(0, self.sz[1])[np.newaxis,:]
        xy[:,:,1] = np.arange(0, self.sz[0])[:,np.newaxis]

        scale = np.random.rand(self.scale_length).astype(np.float32)[0]
        xy_scale_ref = scale * xy

        # upload scale value to GPU constant memory
        scale_ptr = self.module.get_global("g_scale")
        scale_gpu = cp.ndarray((self.scale_length,), cp.float32, scale_ptr)
        scale_gpu[...] = scale

        # allocate GPU memory
        xy_scale_gpu = torch.empty((self.sz[0], self.sz[1], 2), dtype=torch.float32, device='cuda')
        sz_block = 16, 16
        sz_grid = math.ceil(self.sz[1] / sz_block[1]), math.ceil(self.sz[0] / sz_block[0])
        # call the kernel
        self.scale_grid_gpu(
            block=sz_block,
            grid=sz_grid,
            args=(
                xy_scale_gpu.data_ptr(),
            )
        )
        # download the result
        xy_scale = xy_scale_gpu.cpu().numpy()

        err = np.abs(xy_scale - xy_scale_ref)
        ok_(np.max(err) < self.eps)
