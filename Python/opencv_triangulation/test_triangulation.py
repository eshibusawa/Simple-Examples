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

from unittest import TestCase
from nose.tools import ok_

import numpy as np
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

class TriangulationTestCase(TestCase):
    def setUp(self):
        # grid
        x_range = np.arange(0, 120, 10, dtype=np.float32) # [mm]
        y_range = np.arange(0, 100, 10, dtype=np.float32) # [mm]
        x_range -= np.average(x_range)
        y_range -= np.average(y_range)
        xyz = np.empty((3, y_range.shape[0], x_range.shape[0]), dtype=np.float32)
        xyz[0, :, :] = x_range[np.newaxis, :]
        xyz[1, :, :] = y_range[:, np.newaxis]
        xyz[2, :, :] = 5
        self.xyz = xyz.reshape(3, -1)
        # internal calibration
        fx = 640
        fy = fx
        u0 = 320
        v0 = 240
        self.K = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]], dtype=np.float32)
        self.dists = np.array([-5E-1, -5E-1, 0, 0, -5E-1], dtype=np.float32)
        # external calibration
        self.B = 80 # [mm]
        self.t_system = 300 # [mm]
        self.R_system, _ = cv2.Rodrigues(np.deg2rad(15)*np.array([1, 0, 0], dtype=np.float32))
        self.Rl = self.R_system
        self.tl = np.array([-self.B/2, 0, self.t_system], dtype=np.float32)[:,np.newaxis]
        self.Rr = self.R_system
        self.tr = np.array([self.B/2, 0, self.t_system], dtype=np.float32)[:,np.newaxis]
        #
        self.eps_px = 1E-2
        self.eps_mm = 1E-1
        self.enable_write_debug_image = False

    def tearDown(self):
        pass

    def perspective_projection_test(self):
        # project (zero distortion)
        xyz_l = np.dot(self.Rl, self.xyz) + self.tl
        uv_l = np.dot(self.K, xyz_l/xyz_l[2,:])[:2,:]
        xyz_r = np.dot(self.Rr, self.xyz) + self.tr
        uv_r = np.dot(self.K, xyz_r/xyz_r[2,:])[:2,:]

        if self.enable_write_debug_image:
            img_l = draw_grid(uv_l)
            cv2.imwrite('left.png', img_l)
            img_r = draw_grid(uv_r)
            cv2.imwrite('right.png', img_r)

        # left
        zero_dists = np.zeros_like(self.dists)
        uv_l_ref = get_projection(self.K, self.Rl, self.tl, zero_dists, self.xyz)
        err = np.abs(uv_l_ref - uv_l)
        ok_(np.max(err) < self.eps_px)

        # right
        uv_r_ref = get_projection(self.K, self.Rr, self.tr, zero_dists, self.xyz)
        err = np.abs(uv_r_ref - uv_r)
        ok_(np.max(err) < self.eps_px)

    def perspective_triangulation_test(self):
        # left
        zero_dists = np.zeros_like(self.dists)
        uv_l = get_projection(self.K, self.Rl, self.tl, zero_dists, self.xyz)

        # right
        uv_r = get_projection(self.K, self.Rr, self.tr, zero_dists, self.xyz)

        Pl = get_KRt(self.K, self.Rl, self.tl)
        Pr = get_KRt(self.K, self.Rr, self.tr)
        xyz_est_h = cv2.triangulatePoints(Pl, Pr, uv_l, uv_r)
        xyz_est = xyz_est_h[:3, :] / xyz_est_h[3, :]
        sign = xyz_est[2,:] < 0
        xyz_est[:,sign] = -xyz_est[:,sign]

        err = np.abs(self.xyz - xyz_est)
        ok_(np.max(err) < self.eps_mm)

    def undistortion_test(self):
        # left
        uv_l = get_projection(self.K, self.Rl, self.tl, self.dists, self.xyz)
        # right
        uv_r = get_projection(self.K, self.Rr, self.tr, self.dists, self.xyz)

        if self.enable_write_debug_image:
            img_l = draw_grid(uv_l)
            cv2.imwrite('left_distorted.png', img_l)
            img_r = draw_grid(uv_r)
            cv2.imwrite('right_distorted.png', img_r)

        uv_l_undistorted = cv2.undistortPoints(uv_l, self.K, self.dists, P = self.K)
        uv_l_undistorted = uv_l_undistorted[:,0,:].T
        uv_r_undistorted = cv2.undistortPoints(uv_r, self.K, self.dists, P = self.K)
        uv_r_undistorted = uv_r_undistorted[:,0,:].T

        if self.enable_write_debug_image:
            img_l = draw_grid(uv_l_undistorted)
            cv2.imwrite('left_undistorted.png', img_l)
            img_r = draw_grid(uv_r_undistorted)
            cv2.imwrite('right_undistorted.png', img_r)

        # left (zero distortion)
        zero_dists = np.zeros_like(self.dists)
        uv_l_undistorted_ref = get_projection(self.K, self.Rl, self.tl, zero_dists, self.xyz)
        # right (zero distortion)
        uv_r_undistorted_ref = get_projection(self.K, self.Rr, self.tr, zero_dists, self.xyz)

        err = np.abs(uv_l_undistorted_ref - uv_l_undistorted)
        ok_(np.max(err) < self.eps_px)
        err = np.abs(uv_r_undistorted_ref - uv_r_undistorted)
        ok_(np.max(err) < self.eps_px)

    def distorted_triangulation_test(self):
        # left
        uv_l = get_projection(self.K, self.Rl, self.tl, self.dists, self.xyz)
        # right
        uv_r = get_projection(self.K, self.Rr, self.tr, self.dists, self.xyz)

        uv_l_undistorted = cv2.undistortPoints(uv_l, self.K, self.dists, P = self.K)
        uv_l_undistorted = uv_l_undistorted[:,0,:].T
        uv_r_undistorted = cv2.undistortPoints(uv_r, self.K, self.dists, P = self.K)
        uv_r_undistorted = uv_r_undistorted[:,0,:].T

        # triangulation
        Pl = get_KRt(self.K, self.Rl, self.tl)
        Pr = get_KRt(self.K, self.Rr, self.tr)
        xyz_est_h = cv2.triangulatePoints(Pl, Pr, uv_l_undistorted, uv_r_undistorted)
        xyz_est = xyz_est_h[:3, :] / xyz_est_h[3, :]
        sign = xyz_est[2,:] < 0
        xyz_est[:,sign] = -xyz_est[:,sign]

        err = np.abs(self.xyz - xyz_est)
        ok_(np.max(err) < self.eps_mm)
