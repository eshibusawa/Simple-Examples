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

import pytest
from typing import Dict, Any, Generator

import numpy as np
from numpy.testing import assert_allclose
import cv2

def draw_grid(uv):
    width = np.maximum(np.max(uv[0,:]), 640)
    height = np.maximum(np.max(uv[1,:]), 480)
    img = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    for p in uv.T:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 0), 2)
    return img

def get_KRt(K, R, t):
    P = np.empty((3, 4), dtype=K.dtype)
    P[:,:3] = np.dot(K, R)
    P[:,3] = np.dot(K, t)[:,0]
    return P

def get_projection(K, R, t, dists, xyz):
    rvec, _ = cv2.Rodrigues(R)
    tvec = t[:,0]
    uv, _ = cv2.projectPoints(xyz, rvec, tvec, K, dists)
    return uv[:,0,:].T

@pytest.fixture(scope='module')
def setup() -> Generator[Dict[str, Any], Any, None]:
    # grid
    x_range = np.arange(0, 120, 10, dtype=np.float32) # [mm]
    y_range = np.arange(0, 100, 10, dtype=np.float32) # [mm]
    x_range -= np.average(x_range)
    y_range -= np.average(y_range)
    xyz = np.empty((3, y_range.shape[0], x_range.shape[0]), dtype=np.float32)
    xyz[0, :, :] = x_range[np.newaxis, :]
    xyz[1, :, :] = y_range[:, np.newaxis]
    xyz[2, :, :] = 5
    xyz = xyz.reshape(3, -1)
    # internal calibration
    fx = 640
    fy = fx
    u0 = 320
    v0 = 240
    K = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]], dtype=np.float32)
    dists = np.array([-5E-1, -5E-1, 0, 0, -5E-1], dtype=np.float32)
    # external calibration
    B = 80 # [mm]
    t_system = 300 # [mm]
    R_system, _ = cv2.Rodrigues(np.deg2rad(15)*np.array([1, 0, 0], dtype=np.float32))
    Rl = R_system
    tl = np.array([-B/2, 0, t_system], dtype=np.float32)[:,np.newaxis]
    Rr = R_system
    tr = np.array([B/2, 0, t_system], dtype=np.float32)[:,np.newaxis]
    #
    eps_px = 1E-2
    eps_mm = 1E-1
    enable_write_debug_image = False

    yield {
        'eps_px': eps_px,
        'eps_mm': eps_mm,
        'enable_write_debug_image': enable_write_debug_image,
        'xyz': xyz,
        'K': K,
        'dists': dists,
        'Rl': Rl,
        'tl': tl,
        'Rr': Rr,
        'tr': tr,
    }

def test_perspective_projection(setup: Generator[Dict[str, Any], Any, None]) -> None:
    eps_px = setup['eps_px']
    enable_write_debug_image = setup['enable_write_debug_image']
    xyz = setup['xyz']
    K = setup['K']
    dists = setup['dists']
    Rl = setup['Rl']
    tl = setup['tl']
    Rr = setup['Rr']
    tr = setup['tr']

    # project (zero distortion)
    xyz_l = np.dot(Rl, xyz) + tl
    uv_l = np.dot(K, xyz_l/xyz_l[2,:])[:2,:]
    xyz_r = np.dot(Rr, xyz) + tr
    uv_r = np.dot(K, xyz_r/xyz_r[2,:])[:2,:]

    if enable_write_debug_image:
        img_l = draw_grid(uv_l)
        cv2.imwrite('left.png', img_l)
        img_r = draw_grid(uv_r)
        cv2.imwrite('right.png', img_r)

    # left
    zero_dists = np.zeros_like(dists)
    uv_l_ref = get_projection(K, Rl, tl, zero_dists, xyz)
    assert_allclose(uv_l_ref, uv_l, rtol=eps_px)

    # right
    uv_r_ref = get_projection(K, Rr, tr, zero_dists, xyz)
    assert_allclose(uv_r_ref, uv_r, rtol=eps_px)

def test_perspective_triangulation(setup: Generator[Dict[str, Any], Any, None]) -> None:
    eps_mm = setup['eps_mm']
    xyz = setup['xyz']
    K = setup['K']
    dists = setup['dists']
    Rl = setup['Rl']
    tl = setup['tl']
    Rr = setup['Rr']
    tr = setup['tr']

    # left
    zero_dists = np.zeros_like(dists)
    uv_l = get_projection(K, Rl, tl, zero_dists, xyz)

    # right
    uv_r = get_projection(K, Rr, tr, zero_dists, xyz)

    Pl = get_KRt(K, Rl, tl)
    Pr = get_KRt(K, Rr, tr)
    xyz_est_h = cv2.triangulatePoints(Pl, Pr, uv_l, uv_r)
    xyz_est = xyz_est_h[:3, :] / xyz_est_h[3, :]
    sign = xyz_est[2,:] < 0
    xyz_est[:,sign] = -xyz_est[:,sign]
    assert_allclose(xyz, xyz_est, rtol=eps_mm)

def test_undistortion(setup: Generator[Dict[str, Any], Any, None]) -> None:
    eps_px = setup['eps_px']
    enable_write_debug_image = setup['enable_write_debug_image']
    xyz = setup['xyz']
    K = setup['K']
    dists = setup['dists']
    Rl = setup['Rl']
    tl = setup['tl']
    Rr = setup['Rr']
    tr = setup['tr']

    # left
    uv_l = get_projection(K, Rl, tl, dists, xyz)
    # right
    uv_r = get_projection(K, Rr, tr, dists, xyz)

    if enable_write_debug_image:
        img_l = draw_grid(uv_l)
        cv2.imwrite('left_distorted.png', img_l)
        img_r = draw_grid(uv_r)
        cv2.imwrite('right_distorted.png', img_r)

    uv_l_undistorted = cv2.undistortPoints(uv_l, K, dists, P = K)
    uv_l_undistorted = uv_l_undistorted[:,0,:].T
    uv_r_undistorted = cv2.undistortPoints(uv_r, K, dists, P = K)
    uv_r_undistorted = uv_r_undistorted[:,0,:].T

    if enable_write_debug_image:
        img_l = draw_grid(uv_l_undistorted)
        cv2.imwrite('left_undistorted.png', img_l)
        img_r = draw_grid(uv_r_undistorted)
        cv2.imwrite('right_undistorted.png', img_r)

    # left (zero distortion)
    zero_dists = np.zeros_like(dists)
    uv_l_undistorted_ref = get_projection(K, Rl, tl, zero_dists, xyz)
    # right (zero distortion)
    uv_r_undistorted_ref = get_projection(K, Rr, tr, zero_dists, xyz)

    assert_allclose(uv_l_undistorted_ref, uv_l_undistorted, rtol=eps_px)
    assert_allclose(uv_r_undistorted_ref, uv_r_undistorted, rtol=eps_px)

def test_distorted_triangulation(setup: Generator[Dict[str, Any], Any, None]) -> None:
    eps_mm = setup['eps_mm']
    xyz = setup['xyz']
    K = setup['K']
    dists = setup['dists']
    Rl = setup['Rl']
    tl = setup['tl']
    Rr = setup['Rr']
    tr = setup['tr']

    # left
    uv_l = get_projection(K, Rl, tl, dists, xyz)
    # right
    uv_r = get_projection(K, Rr, tr, dists, xyz)

    uv_l_undistorted = cv2.undistortPoints(uv_l, K, dists, P = K)
    uv_l_undistorted = uv_l_undistorted[:,0,:].T
    uv_r_undistorted = cv2.undistortPoints(uv_r, K, dists, P = K)
    uv_r_undistorted = uv_r_undistorted[:,0,:].T

    # triangulation
    Pl = get_KRt(K, Rl, tl)
    Pr = get_KRt(K, Rr, tr)
    xyz_est_h = cv2.triangulatePoints(Pl, Pr, uv_l_undistorted, uv_r_undistorted)
    xyz_est = xyz_est_h[:3, :] / xyz_est_h[3, :]
    sign = xyz_est[2,:] < 0
    xyz_est[:,sign] = -xyz_est[:,sign]
    assert_allclose(xyz, xyz_est, rtol=eps_mm)
