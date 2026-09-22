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
#  Mirror eyes application example

from __future__ import annotations

import logging

import cv2

# own
from mirror_eyes.config.config import IRIS_COLORS, Config
from mirror_eyes.custom_types import Limits
from mirror_eyes.mirror_eyes import MirrorEyes
from mirror_eyes.tools.drawing import image_filters
from mirror_eyes_app.attention import Attention
from mirror_eyes_app.camera import Camera

#from mirror_eyes_app.attention import FaceFollower as AttentionModule
from mirror_eyes_app.selective_attention import SelectiveFaceTracker as AttentionModule

colornames = list(IRIS_COLORS.keys())

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(filename="debug.log", mode="w"),  # mode w to override previous runs
        logging.StreamHandler(),
    ],
)


class App:
    """The App class integrates mirror eyes, attention and an image source."""

    def __init__(self, mirror_eyes: MirrorEyes = None, img_source: Camera = None, attention: Attention = None):
        self.mirror_eyes = mirror_eyes
        self.mirror_eyes.setup()
        self.img_source = img_source
        self.attention = attention
        self.active = False
        # eye position definition:
        # [x, y, z, focus_distance] specified in meters, relative to the center of the canvas
        self.eye_positions = [[-0.06, 0.00, 0.0, 1], [0.06, 0.00, 0.0, 1]]
        self.color_idx = 0
        self.opacity = mirror_eyes.cfg.eyes[0].mirror.opacity.opacity_range.min#  0.6
        self.filter_idx = 0

    def process_image_for_attention(self, origimage: cv2.typing.MatLike):
        """only use if required.
        Carries out image operations that make an image compatible with the attention module.
        """
        processed_image = cv2.flip(cv2.cvtColor(origimage, cv2.COLOR_BGR2RGB), 1)  # SUBSTITUTE/EXTEND
        return processed_image

    def process_keyboard_input(self):
        """button-based control of application behavior.
        Exemplifies a range of available actions that may be triggered
        on top of mirroring."""
        keys = cv2.waitKey(1) & 0xFF
        # The escape key deactivates the main loop
        if keys == 27:
            self.active = False
        # examples for using the keyboard to control eye behavior
        elif keys == ord("s"):  # smiling
            self.mirror_eyes.start_smiling(max_duration=1)
        elif keys == ord("a"):  # angry
            self.mirror_eyes.close_eyes_at_angle(angle=15, max_duration=1)
        elif keys == ord("h"):
            self.mirror_eyes.start_crying(max_duration=1)
        elif keys == ord("f"):
            self.mirror_eyes.flirt(max_duration=1)
        elif keys == ord("r"):
            duration = 1
            self.mirror_eyes.start_roi_scanning(
                max_duration=duration, focus_point_duration_range=Limits(0.2, duration * 0.75)
            )
        elif keys == ord("l"):
            if self.mirror_eyes.showing_loading_animation():
                self.mirror_eyes.stop_loading_animation()
            else:
                self.mirror_eyes.show_loading_animation()
        elif keys == ord("c"):
            self.color_idx += 1
            if self.color_idx >= len(colornames):
                self.color_idx = 0
            self.mirror_eyes.change_eye_color(color=IRIS_COLORS[colornames[self.color_idx]], duration=0.3)
        elif keys == ord("m"):
            self.filter_idx += 1
            if self.filter_idx >= len(image_filters):
                self.filter_idx = 0
            self.mirror_eyes.change_reflection_filter(image_filters[self.filter_idx])
            # self.image_filter = image_filters[self.filter_idx]
        elif keys == ord("+"):
            self.mirror_eyes.update_pupil_size_by(delta=5)
        elif keys == ord("-"):
            self.mirror_eyes.update_pupil_size_by(delta=-5)
        elif keys == ord("y"):
            self.opacity = max(self.opacity - 0.1, -1.0)
        elif keys == ord("x"):
            self.opacity = min(self.opacity + 0.1, 1.5)
        # switch between tracked targets using number keys
        elif keys == ord("1"):
            self.attention.target_idx = 0
        elif keys == ord("2"):
            self.attention.target_idx = 1
        elif keys == ord("3"):
            self.attention.target_idx = 2
        elif keys == ord("0"):
            self.attention.target_idx = -1
        # only for testing in external control mode: move eye positions left or right
        elif keys == ord(","):
            self.eye_positions[0][0] += 0.005
            self.eye_positions[1][0] += 0.005
        elif keys == ord("."):
            self.eye_positions[0][0] -= 0.005
            self.eye_positions[1][0] -= 0.005

    def update(self):
        self.process_keyboard_input()
        #if self.img_source.new_frame:  # SUBSTITUTE <- unnecessary slow-down
        new_frame = self.img_source.origimage  # SUBSTITUTE
        self.attention.update(self.process_image_for_attention(new_frame))  # SUBSTITUTE

        self.attention.apply_scale_offset(
            self.mirror_eyes.img_scale_based_offset)

        self.mirror_eyes.update(
            img_frame=new_frame,
            poi=self.attention.poi,  # requires coordinates in the form [screen_px_x, screen_px_y, focus_distance]
            opacity=self.opacity,
            eye_positions=self.eye_positions,  # self.attention.eye_positions
            roi=self.attention.poi_bounding_box,
        )
        self.img_source.new_frame = False

    def loop(self):
        logging.info("Starting application loop")
        self.active = True
        while self.active:
            self.update()
        logging.info("Closing application loop")

    def shutdown(self):
        self.active = False
        self.attention.shutdown()
        self.img_source.shutdown()
        self.mirror_eyes.shutdown()

    def run(self):
        """Start application loop and trigger
        the shutdown procedure when it is over"""
        try:
            self.loop()
        except KeyboardInterrupt:
            self.active = False
        finally:
            self.shutdown()


def main():
    Config().set_config("local_eye_control")  #'external_eye_control')
    cfg = Config().get_config()

    # camera setup
    camera = Camera(
        width=cfg.camera.resolution.width,
        height=cfg.camera.resolution.height,
        idx=cfg.camera.idx,
        img_scale=cfg.camera.img_scale,
    )
    camera.start_capture()

    # gaze selection setup
    attention = AttentionModule(
        width=camera.width, height=camera.height, xlim=(0, camera.width), ylim=(0, camera.height)
    )
    attention.target_idx = 1

    # mirror eyes setup
    me = MirrorEyes(config=cfg)
    # optional: setup video recording
    if cfg.display.record_video:
        me.video_recorder.active = True
        me.video_recorder.recorder_setup()

    app = App(mirror_eyes=me, img_source=camera, attention=attention)
    app.run()


if __name__ == "__main__":
    #import yappi
    #yappi.set_clock_type("wall")
    #yappi.start()
    try:
        main()
    finally:
        #yappi.stop()
        #stats = yappi.get_func_stats()
        #stats.sort("ttot").print_all()  # sort by total time
        logging.info('Bye')

