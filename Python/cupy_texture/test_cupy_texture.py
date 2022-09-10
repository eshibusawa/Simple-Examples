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
import cv2

class CupyTextureTestCase(TestCase):
    def setUp(self):
        self.eps = 2

        dn = os.path.dirname(os.path.realpath(__file__))
        fpfn = os.path.join(dn, 'cupy_texture.cu')
        # load raw kernel
        with open(fpfn, 'r') as f:
            cuda_source = f.read()
        cuda_source = cuda_source.replace('TEXUTURE_TEST_PIXEL_TYPE', 'unsigned char')
        self.module = cp.RawModule(code=cuda_source)
        self.copy_texture = self.module.get_function("copyTexture")

    def tearDown(self):
        pass

    def bindless_texture_test(self):
        array_in_cpu = (255 * np.random.rand(255, 255)).astype(np.uint8)
        array_in = cp.array(array_in_cpu, dtype=cp.uint8)
        array_out = cp.zeros_like(array_in)

        assert array_in.flags.c_contiguous
        assert array_out.flags.c_contiguous

        channel_format_descriptor = cp.cuda.texture.ChannelFormatDescriptor(8, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindUnsigned)
        array_in_2d = cp.cuda.texture.CUDAarray(channel_format_descriptor, array_in.shape[1], array_in.shape[0])
        array_in_2d.copy_from(array_in)

        resouce_descriptor = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray,
            cuArr = array_in_2d)
        texture_descriptor = cp.cuda.texture.TextureDescriptor(addressModes = (cp.cuda.runtime.cudaAddressModeBorder, cp.cuda.runtime.cudaAddressModeBorder),
            filterMode=cp.cuda.runtime.cudaFilterModePoint,
            readMode=cp.cuda.runtime.cudaReadModeElementType,
            normalizedCoords = 0)
        texuture_object = cp.cuda.texture.TextureObject(resouce_descriptor, texture_descriptor)
        sz_block = 16, 16
        sz_grid = math.ceil(array_out.shape[1] / sz_block[0]), math.ceil(array_out.shape[0] / sz_block[1])
        # call the kernel
        self.copy_texture(
            block=sz_block,
            grid=sz_grid,
            args=(
                array_out,
                texuture_object,
                array_out.shape[1],
                array_out.shape[0]
            )
        )

        err = np.abs(array_in_cpu - cp.asnumpy(array_out))
        ok_(np.max(err) < self.eps)

class CupyRGBATextureTestCase(TestCase):
    def setUp(self):
        self.eps = 2

        dn = os.path.dirname(os.path.realpath(__file__))
        fpfn = os.path.join(dn, 'cupy_texture.cu')
        # load raw kernel
        with open(fpfn, 'r') as f:
            cuda_source = f.read()
        cuda_source = cuda_source.replace('TEXUTURE_TEST_PIXEL_TYPE', 'uchar4')
        self.module = cp.RawModule(code=cuda_source)
        self.copy_texture = self.module.get_function("copyTexture")

    def tearDown(self):
        pass

    @staticmethod
    def get_color_chart(sz):
        xy = np.empty((sz[0], sz[1], 2), dtype=np.float32)
        xy[:,:,0] = np.arange(0, sz[1])[np.newaxis,:] - sz[1]/2
        xy[:,:,1] = np.arange(0, sz[0])[:,np.newaxis] - sz[0]/2

        xy_rgb = np.empty((xy.shape[0], xy.shape[1], 4), dtype=np.uint8)
        dir = np.arctan2(xy[:,:, 1], xy[:,:,0])
        xy_rgb[:,:,2] = 127 * (np.sin(dir) + 1)
        xy_rgb[:,:,1] = 127 * (np.cos(dir) + 1)
        xy_rgb[:,:,0] = 127
        xy_rgb[:,:,3] = 255

        return xy_rgb

    def bindless_texture_test(self):
        array_in_cpu = self.get_color_chart((255, 255))
        array_in = cp.array(array_in_cpu, dtype=cp.uint8)
        array_out = cp.zeros_like(array_in)

        assert array_in.flags.c_contiguous
        assert array_out.flags.c_contiguous

        channel_format_descriptor = cp.cuda.texture.ChannelFormatDescriptor(8, 8, 8, 8, cp.cuda.runtime.cudaChannelFormatKindUnsigned)
        array_in_2d = cp.cuda.texture.CUDAarray(channel_format_descriptor, array_in.shape[1], array_in.shape[0])
        array_in_2d.copy_from(array_in.reshape(array_in_cpu.shape[0], -1))

        resouce_descriptor = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray,
            cuArr = array_in_2d)
        texture_descriptor = cp.cuda.texture.TextureDescriptor(addressModes = (cp.cuda.runtime.cudaAddressModeBorder, cp.cuda.runtime.cudaAddressModeBorder),
            filterMode=cp.cuda.runtime.cudaFilterModePoint,
            readMode=cp.cuda.runtime.cudaReadModeElementType,
            normalizedCoords = 0)
        texuture_object = cp.cuda.texture.TextureObject(resouce_descriptor, texture_descriptor)
        sz_block = 16, 16
        sz_grid = math.ceil(array_out.shape[1] / sz_block[0]), math.ceil(array_out.shape[0] / sz_block[1])
        # call the kernel
        self.copy_texture(
            block=sz_block,
            grid=sz_grid,
            args=(
                array_out,
                texuture_object,
                array_out.shape[1],
                array_out.shape[0]
            )
        )

        err = np.abs(array_in_cpu - cp.asnumpy(array_out))
        ok_(np.max(err) < self.eps)

class CupySphereMaskTestCase(TestCase):
    def setUp(self):
        self.eps = 2

        dn = os.path.dirname(os.path.realpath(__file__))
        fpfn = os.path.join(dn, 'cupy_texture.cu')
        # load raw kernel
        with open(fpfn, 'r') as f:
            cuda_source = f.read()
        cuda_source = cuda_source.replace('TEXUTURE_TEST_PIXEL_TYPE', 'uchar4')
        self.module = cp.RawModule(code=cuda_source)
        self.copy_texture = self.module.get_function("copyTextureMasked")

    def tearDown(self):
        pass

    def sphere_mask_est(self):
        array_in_cpu = CupyRGBATextureTestCase.get_color_chart((255, 255))
        array_in = cp.array(array_in_cpu, dtype=cp.uint8)
        array_out = cp.zeros_like(array_in)

        mask_cpu = np.zeros(array_in_cpu.shape[:2], dtype = np.uint8)
        cv2.circle(mask_cpu, (mask_cpu.shape[1]//2, mask_cpu.shape[0]//2), mask_cpu.shape[0]//2, 255, -1)
        xy = np.empty((mask_cpu.shape[0], mask_cpu.shape[1], 2), dtype=np.uint32)
        xy[:,:,0] = np.arange(0, mask_cpu.shape[1])[np.newaxis,:]
        xy[:,:,1] = np.arange(0, mask_cpu.shape[0])[:,np.newaxis]
        mask_xy_cpu = xy[mask_cpu != 0]
        mask_xy = cp.array(mask_xy_cpu[np.newaxis,:,:])

        assert array_out.flags.c_contiguous

        channel_format_descriptor = cp.cuda.texture.ChannelFormatDescriptor(8, 8, 8, 8, cp.cuda.runtime.cudaChannelFormatKindUnsigned)
        array_in_2d = cp.cuda.texture.CUDAarray(channel_format_descriptor, array_in.shape[1], array_in.shape[0])
        array_in_2d.copy_from(array_in.reshape(array_in_cpu.shape[0], -1))

        resouce_descriptor = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray,
            cuArr = array_in_2d)
        texture_descriptor = cp.cuda.texture.TextureDescriptor(addressModes = (cp.cuda.runtime.cudaAddressModeBorder, cp.cuda.runtime.cudaAddressModeBorder),
            filterMode=cp.cuda.runtime.cudaFilterModePoint,
            readMode=cp.cuda.runtime.cudaReadModeElementType,
            normalizedCoords = 0)
        texuture_object = cp.cuda.texture.TextureObject(resouce_descriptor, texture_descriptor)

        channel_format_descriptor_mask = cp.cuda.texture.ChannelFormatDescriptor(32, 32, 0, 0, cp.cuda.runtime.cudaChannelFormatKindUnsigned)
        mask_xy_2d = cp.cuda.texture.CUDAarray(channel_format_descriptor_mask, mask_xy.shape[1], mask_xy.shape[0])
        mask_xy_2d.copy_from(mask_xy.reshape(mask_xy.shape[0], -1))

        resouce_descriptor_mask = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray,
            cuArr = mask_xy_2d)
        texture_descriptor_mask = cp.cuda.texture.TextureDescriptor(addressModes = (cp.cuda.runtime.cudaAddressModeBorder, cp.cuda.runtime.cudaAddressModeBorder),
            filterMode=cp.cuda.runtime.cudaFilterModePoint,
            readMode=cp.cuda.runtime.cudaReadModeElementType,
            normalizedCoords = 0)
        texuture_object_mask = cp.cuda.texture.TextureObject(resouce_descriptor_mask, texture_descriptor_mask)

        sz_block = 1024, 1
        sz_grid = math.ceil(mask_xy.shape[1] / sz_block[0]), 1
        # call the kernel
        self.copy_texture(
            block=sz_block,
            grid=sz_grid,
            args=(
                array_out,
                texuture_object,
                texuture_object_mask,
                mask_xy.shape[1],
                array_out.shape[1],
                array_out.shape[0]
            )
        )

        err = np.abs(array_in_cpu[mask_cpu != 0] - cp.asnumpy(array_out[mask_cpu != 0]))
        ok_(np.max(err) < self.eps)
