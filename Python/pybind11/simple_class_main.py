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

import numpy as np

from simple_pybind import simple_class
from simple_pybind import get_class_a
from simple_pybind import get_class_b

class class_a:
    def __init__(self):
        self.value = 1
        self.computation_mode = simple_class.computation_mode.add

class class_b:
    def __init__(self):
        self.value = -1
        self.computation_mode = simple_class.computation_mode.sub

if __name__ == '__main__':
    arr = np.random.rand(3, 3).astype(np.float32)
    sc = simple_class()
    arr2 = sc.compute(class_a(), arr)
    print('err: {}'.format(np.max(np.abs(arr2 - (arr + 1)))))
    arr2 = sc.compute(class_b(), arr)
    print('err: {}'.format(np.max(np.abs(arr2 - (arr - (-1))))))

    arr2 = sc.compute(get_class_a(), arr)
    print('err: {}'.format(np.max(np.abs(arr2 - (arr + 1)))))
    arr2 = sc.compute(get_class_b(), arr)
    print('err: {}'.format(np.max(np.abs(arr2 - (arr - (-1))))))
