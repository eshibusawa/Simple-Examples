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

from PySide6.QtCore import Slot, QMetaObject, QThread
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog

import cv2

from camera_capture import camera_capture as cp
from camera_ui import Ui_Dialog as CameraDialogUI

class CameraDialog(QDialog):
    def __init__(self, parent=None):
        super(CameraDialog, self).__init__(parent)
        self.ui = CameraDialogUI()
        self.ui.setupUi(self)

        self.capture = cp()
        self.capture.initialize(0)
        self.capture.done_capture.connect(self.update)
        self.ui.pbStart.clicked.connect(self.start_capture)
        self.ui.pbStop.clicked.connect(self.stop_capture)

        self.capture_thread = QThread(self)
        self.capture.moveToThread(self.capture_thread)
        self.capture_thread.start()

    def stop_thread(self):
        self.capture.set_stop_flag(True)
        while self.capture_thread.isRunning():
            self.capture_thread.quit()
            self.capture_thread.wait()

    def __del__(self):
        self.stop_thread()

    @Slot()
    def update(self):
        fr = self.capture.get_data()
        if fr is not None:
            sz = self.ui.lFrame.size()
            h, w = sz.height(), sz.width()
            fr2 = cv2.resize(fr, dsize=(w, h))
            qi = QImage(fr2.flatten(), w, h, QImage.Format_BGR888)
            self.ui.lFrame.setPixmap(QPixmap.fromImage(qi))

    @Slot()
    def start_capture(self):
        self.ui.pbStart.setEnabled(False)
        self.capture.set_stop_flag(False)
        QMetaObject.invokeMethod(self.capture, 'capture_loop')
        self.ui.pbStop.setEnabled(True)

    @Slot()
    def stop_capture(self):
        self.ui.pbStop.setEnabled(False)
        self.capture.set_stop_flag(True)
        self.ui.pbStart.setEnabled(True)
