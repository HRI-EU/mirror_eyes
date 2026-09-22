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

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class VideoRecorder:
    def __init__(self, height, width, fps=30, name="Video_capture", active=False, directory=None):
        self.height = height
        self.width = width
        self.name = name
        self.fps = fps
        self.active = active
        if directory is None:
            self.directory = Path.cwd()
        else:
            self.directory = directory
        if active:
            self.recorder_setup()

    def recorder_setup(self):
        self.recording_fn = os.path.join(
            self.directory, f"{self.name}_{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.avi"
        )
        self.recording = cv2.VideoWriter(
            self.recording_fn, cv2.VideoWriter_fourcc("M", "J", "P", "G"), self.fps, (self.width, self.height)
        )
        self.video_frame = np.ones((self.height, self.width, 3), dtype="uint8") * 255
        logging.info(
            f"Recording {self.width}x{self.height} video\
              to {self.recording_fn} at {self.fps} fps"
        )

    def update_video_capture(self):
        """Add current frame to video recording"""
        self.recording.write(self.video_frame)

    def shutdown(self):
        if self.active:
            self.recording.release()
            logging.info(f"Stored recording in {self.recording_fn}")


class SingleScreenMirrorEyesRecorder(VideoRecorder):
    """Extension of basic VideoRecorder class for the purpose
    of storing a run of a single screen MirrorEyes instance as a video file

    Args:
        eye (EyeModel) - eye model object containing width and height
        name (str) - name/label of the capture procedure
            used as filename prefix
        active (bool) - set to True to record, False to not record
        num_eyes (int) - presently 1 or two are functional (experimental)
        fps (int) - framerate of the recorded video
    """

    def __init__(self, canvas, name="Mirror_eye_capture", active=False, fps=30, directory=None):
        self.name = name
        self.fps = fps
        self.active = active
        # self.setup_parameters_based_on_one_eye(eye, relative_distance_between_eyes=0.6)
        self.setup_parameters_based_on_canvas(canvas)
        if directory is None:
            self.directory = Path.cwd()
        else:
            self.directory = directory
        if active:
            self.recorder_setup()

    def setup_parameters_based_on_canvas(self, canvas):
        height, width, _channels = canvas.shape
        self.height = height
        self.width = width

    def update_video_capture_from_canvas(self, canvas):
        if self.active:
            self.video_frame[0 : self.height, 0 : self.width] = canvas
            self.update_video_capture()
