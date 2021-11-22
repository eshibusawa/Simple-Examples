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

class CupyTextureTestCase(TestCase):
    def setUp(self):
        self.eps = 2

        dn = os.path.dirname(os.path.realpath(__file__))
        fpfn = os.path.join(dn, 'cupy_texture.cu')
        # load raw kernel
        with open(fpfn, 'r') as f:
            cuda_source = f.read()
        self.module = cp.RawModule(code=cuda_source)
        self.copy_texture = self.module.get_function("copyTexture")

    def tearDown(self):
        pass

    def bindress_texture_test(self):
        array_in_cpu = (255 * np.random.rand(255, 255)).astype(np.uint8)
        array_in = cp.array(array_in_cpu, dtype=cp.uint8)
        array_out = cp.zeros(array_in_cpu.shape, dtype=cp.uint8)

        assert array_in.flags.c_contiguous
        assert array_out.flags.c_contiguous

        channel_format_descriptor = cp.cuda.texture.ChannelFormatDescriptor(8, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindUnsigned)
        array_in_2d = cp.cuda.texture.CUDAarray(channel_format_descriptor, array_in.shape[1], array_in.shape[0])
        array_in_2d.copy_from(array_in)

        resouce_descriptor = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray,
            cuArr = array_in_2d)
        texture_descriptor = cp.cuda.texture.TextureDescriptor(addressModes = (cp.cuda.runtime.cudaAddressModeWrap, cp.cuda.runtime.cudaAddressModeWrap),
            filterMode=cp.cuda.runtime.cudaFilterModePoint,
            readMode=cp.cuda.runtime.cudaReadModeElementType,
            normalizedCoords = 0)
        texuture_object = cp.cuda.texture.TextureObject(resouce_descriptor, texture_descriptor)
        sz_block = 16, 16
        sz_grid = math.ceil(array_out.shape[1] / sz_block[1]), math.ceil(array_out.shape[0] / sz_block[0])
        # call the kernel
        self.copy_texture(
            block=sz_block,
            grid=sz_grid,
            args=(
                array_out,
                texuture_object,
                array_in.shape[1],
                array_in.shape[0]
            )
        )

        err = np.abs(array_in_cpu - cp.asnumpy(array_out))
        ok_(np.max(err) < self.eps)
