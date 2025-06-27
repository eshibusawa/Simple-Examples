# BSD 2-Clause License
#
# Copyright (c) 2025, Eijiro SHIBUSAWA
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
import numpy as np
import cupy as cp
import pytest
from numpy.testing import assert_allclose, assert_array_equal

@pytest.fixture(scope='module')
def custom_pod_module_fixture():
    num_pod = 2
    dn = os.path.dirname(os.path.realpath(__file__))
    fpfn = os.path.join(dn, 'cupy_custom_pod.cu')
    with open(fpfn, 'r') as f:
        cuda_source = f.read()
    cuda_source = cuda_source.replace('CUSTOM_POD_NUMBER', str(num_pod))

    module = cp.RawModule(code=cuda_source)
    module.compile()

    yield module, num_pod

def test_upload_custom_pod_to_constant(custom_pod_module_fixture):
    module, num_pod = custom_pod_module_fixture
    eps = 1E-5

    dtype_pod = np.dtype({'x': (np.int32, 0), 'y': (np.int32, 4), 'value': (np.float32, 8)})
    pod_ref = np.empty((num_pod,), dtype=dtype_pod)
    pod_ref[0]['x'] = 1
    pod_ref[0]['y'] = 2
    pod_ref[0]['value'] = 3.45
    pod_ref[1]['x'] = 6
    pod_ref[1]['y'] = 7
    pod_ref[1]['value'] = 8.90

    sz_gpu = cp.empty(1, dtype=cp.int32)
    gpu_func_get_size = module.get_function('getPODSize')
    gpu_func_get_size(
        block=(1,),
        grid=(1,),
        args=(sz_gpu,)
    )
    cp.cuda.runtime.deviceSynchronize()
    sz = int(sz_gpu[0].get())
    assert sz == dtype_pod.itemsize, \
        'Expected POD size {dtype_pod.itemsize}, but got {sz}'

    pod_ptr = module.get_global('g_POD')
    pod_gpu_constant = cp.ndarray((num_pod * sz,), dtype=cp.byte, memptr=pod_ptr)
    pod_gpu_constant[:] = cp.frombuffer(pod_ref.tobytes(), dtype=np.byte)

    pod_gpu = cp.empty((num_pod,), dtype=dtype_pod)
    assert pod_gpu.flags.c_contiguous, 'pod_gpu must be C-contiguous'

    gpu_func_get_pod = module.get_function('getPOD')
    gpu_func_get_pod(
        block=(num_pod,),
        grid=(1,),
        args=(pod_gpu,)
    )
    cp.cuda.runtime.deviceSynchronize()
    pod_result = pod_gpu.get()

    assert_array_equal(pod_ref['x'], pod_result['x'],
                       err_msg="Mismatch in 'x' field")
    assert_array_equal(pod_ref['y'], pod_result['y'],
                       err_msg="Mismatch in 'y' field")
    assert_allclose(pod_ref['value'], pod_result['value'], rtol=eps, atol=0,
                    err_msg="Mismatch in 'value' field")
