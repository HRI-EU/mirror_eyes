#!/usr/bin/env python
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Honda Research Institute Europe GmbH
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
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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
#
# SPDX-License-Identifier: BSD-3-Clause#
#
#  Cv2-based Camera module as an example image source for mirror eyes


import logging
import sys
from threading import Thread

import cv2


class Camera:
    """
    Webcam-based video capture with updating in
    a separate thread. A camera instance can provide
    image input to MirrorEyes.
    """

    def __init__(self, width=1024, height=576, idx=0, img_scale=1):
        self.width = width
        self.height = height
        self.capture_stopped = False
        self.success = self.new_frame = False
        self.origimage = None
        self.img_scale = img_scale
        if isinstance(idx, str):
            # assume kinect is specified as path
            self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # poke kinect into streaming
        else:
            if sys.platform == "win32":
                self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # self.cap.set(cv2.CAP_PROP_FPS, 30)
        if self.cap.isOpened():
            self.width = int(img_scale * self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(img_scale * self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"camera resolution {self.width}x{self.height}")
        else:
            logging.warning(f"Camera {idx} not ready")

    def start_capture(self):
        """run update_camera_image in thread"""
        self.success, self.origimage = self.cap.read()

        self.capture_stopped = False
        self.new_frame = True
        logging.info("Starting camera capture thread")
        # start the thread to read frames from the video stream
        Thread(target=self.update_camera_image, args=()).start()
        return self

    def update_camera_image(self):
        """Camera reading loop, separated from main thread to increase fps"""
        while True:
            if self.capture_stopped:
                return
            (self.success, origimage) = self.cap.read()
            if self.img_scale != 1:
                self.origimage = cv2.resize(origimage, dsize=(self.width, self.height))
            else:
                self.origimage = origimage
            self.new_frame = self.success

    def shutdown(self):
        """Camera shutdown procedure terminates thread and releases the camera"""
        self.capture_stopped = True
        self.cap.release()

