import pytest
from typing import Dict, Any, Generator

import numpy as np
from numpy.testing import assert_allclose

import util_array as ua
import vector_container

@pytest.fixture(scope='module')
def setup() -> Generator[Dict[str, Any], Any, None]:
    eps = 1E-7

    yield {
        'eps': eps,
    }

def test_from_array(setup: Generator[Dict[str, Any], Any, None]) -> None:
    eps = setup['eps']

    vc = vector_container.vector_container()
    arr_ref = np.array(vc.get(), dtype=np.float32)
    arr = np.arange(0, arr_ref.shape[0], dtype=arr_ref.dtype)
    assert_allclose(arr_ref, arr, rtol=eps, atol=0)

    d = vc.get_ptr()
    ptr = d["ptr"]
    arr_rand_ref = np.random.rand(arr_ref.shape[0]).astype(arr_ref.dtype)
    ua.copy_from_array(arr_rand_ref, ptr)
    arr_rand = np.array(vc.get(), dtype=arr_ref.dtype)

    assert_allclose(arr_rand_ref, arr_rand, rtol=eps, atol=0)


def test_to_array(setup: Generator[Dict[str, Any], Any, None]) -> None:
    eps = setup['eps']

    vc = vector_container.vector_container()
    arr_ref = np.array(vc.get(), dtype=np.float32)

    d = vc.get_ptr()
    ptr = d["ptr"]
    arr = np.random.rand(arr_ref.shape[0]).astype(arr_ref.dtype)
    ua.copy_to_array(arr, ptr)
    assert_allclose(arr_ref, arr, rtol=eps, atol=0)
