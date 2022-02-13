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

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, QMetaObject, QReadWriteLock, QThread

class camera_capture(QObject):
    done_capture = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.capture = None
        self.frame = None
        self.stopped = True
        self.lock = QReadWriteLock()

    def __del__(self):
        if self.capture is not None:
            self.capture.release()

    def initialize(self, id):
        if self.capture is None:
            self.capture = cv2.VideoCapture(id)
        self.stopped = False

    @Slot()
    def capture_loop(self):
        while True:
            ret, frame = self.capture.read()
            if ret:
                self.lock.lockForWrite()
                stopped = self.stopped
                self.frame = np.copy(frame)
                self.lock.unlock()
                self.done_capture.emit()
            if stopped:
                break

    def get_data(self):
        frame = None
        self.lock.lockForRead()
        if self.frame is not None:
            frame = np.copy(self.frame)
        self.lock.unlock()
        return frame

    def set_stop_flag(self, flag):
        self.lock.lockForWrite()
        self.stopped = flag
        self.lock.unlock()

if __name__ == '__main__':
    import sys
    from PySide6.QtCore import QCoreApplication
    app = QCoreApplication(sys.argv)

    cap = camera_capture()
    cap.initialize(0)

    capture_thread = QThread()
    cap.moveToThread(capture_thread)
    QMetaObject.invokeMethod(cap, 'capture_loop')
    capture_thread.start()

    while True:
        fr = cap.get_data()
        if fr is not None:
            cv2.imshow('multi threaded capture', fr)
            k = cv2.waitKey(5) & 0xFF
            if k == ord('q'):
                break

    cap.set_stop_flag(True)
    while capture_thread.isRunning():
        capture_thread.quit()
        capture_thread.wait()
