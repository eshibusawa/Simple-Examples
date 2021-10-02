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

from unittest import TestCase
from nose.tools import ok_

import numpy as np

from simple_boost import simple_class

class simple_class_settings:
    def __init__(self):
        self.value = 1
        self.computation_mode = simple_class.computation_mode.add

class SimpleClassTestCase(TestCase):
    def setUp(self):
        self.eps = 10E-5
        self.arr = np.random.rand(3, 3).astype(np.float32)


    def tearDown(self):
        pass

    def add_test(self):
        scs = simple_class_settings()
        scs.value = np.random.rand(1).astype(np.float32)[0]
        scs.computation_mode = simple_class.computation_mode.add

        sc = simple_class()
        arr2 = sc.compute(scs, self.arr)

        arr2_ref = self.arr + scs.value
        err = np.max(np.abs(arr2_ref - arr2))

        ok_(err < self.eps)

    def sub_test(self):
        scs = simple_class_settings()
        scs.value = np.random.rand(1).astype(np.float32)[0]
        scs.computation_mode = simple_class.computation_mode.sub

        sc = simple_class()
        arr2 = sc.compute(scs, self.arr)

        arr2_ref = self.arr - scs.value
        err = np.max(np.abs(arr2_ref - arr2))

        ok_(err < self.eps)

    def mul_test(self):
        scs = simple_class_settings()
        scs.value = np.random.rand(1).astype(np.float32)[0]
        scs.computation_mode = simple_class.computation_mode.mul

        sc = simple_class()
        arr2 = sc.compute(scs, self.arr)

        arr2_ref = self.arr * scs.value
        err = np.max(np.abs(arr2_ref - arr2))

        ok_(err < self.eps)

    def div_test(self):
        scs = simple_class_settings()
        scs.value = np.random.rand(1).astype(np.float32)[0]
        scs.computation_mode = simple_class.computation_mode.div

        sc = simple_class()
        arr2 = sc.compute(scs, self.arr)

        arr2_ref = self.arr / scs.value
        err = np.max(np.abs(arr2_ref - arr2))

        ok_(err < self.eps)
