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

# standard
from __future__ import annotations

import logging
import math
from time import time

# external
import cv2
import numpy as np
from numpy import ndarray

# own
import mirror_eyes.config.config as cfg
from mirror_eyes.custom_types import Limits, ScreenPoint
from mirror_eyes.tools.color import TransitioningColor
from mirror_eyes.tools.drawing import (
    LoadingCircle,
    RotatingImage,
    draw_line_with_corners,
    line_through_center_at_angle,
    make_blur_circle,
)


class Iris:
    """Iris object used by EyeModel.
    Contains functionality for construction, plotting, and border checking.
    """

    def __init__(self, iris_cfg: cfg.IrisConfig | None = None, img_file: ndarray | None = None) -> None:
        """Args:
        ----
          iris_cfg (IrisConfig): configuration class containing details about color, radius, visibility, style, etc.
        """
        self.cfg = iris_cfg
        self.radius = iris_cfg.radius
        self.width = self.height = self.radius * 2
        self.center = ScreenPoint(x=self.width // 2, y=self.height // 2)
        self._color = TransitioningColor(*iris_cfg.color)
        self.visible = iris_cfg.visible
        self.style = iris_cfg.style

        self.rotating_styles = ["beams"]
        self.rotating_image = RotatingImage()
        self.loading_circle = LoadingCircle(
            radius=self.radius, lines=32, cycle_duration=2.0, brightness_limits=Limits(min=0, max=80)
        )
        self.loading_animation_active = False

        # if no image filepath is provided the iris will be drawn
        if img_file is None:
            self.build_img()  # color=color, radius=self.radius)
            self.alpha_l = self.img[:, :, 3] / 255.0
            self.alpha_s = 1.0 - self.alpha_l
        # otherwise an image is loaded
        else:
            self.load_img(img_file, self.radius)
            self.alpha_s = self.img[:, :, 3] / 255.0
            self.alpha_l = 1.0 - self.alpha_s
        self.blur_circle = make_blur_circle(radius=int(self.radius-0.25*self.radius), blurwidth=self.cfg.blur_width)

        # init with arbitrary values
        # this should be called externally with appropriate limits
        self.set_position_limits_in_eye(self.radius * 4, self.radius * 4)
        self.border_coords = self.center_position_to_border_coordinates(self.radius * 2, self.radius * 2)
        logging.info(f"Iris initialized with size {self.width}x{self.height} px and color {self.color}")
        self._smiling = False
        self._crying = False
        self._eye_closed = False
        self._eye_closed_at_angle = False
        self.eye_closed_angle = 0
        self.max_smiletime = 2  # seconds
        self.max_crytime = 2
        self.max_eye_closetime = 2
        self.smile_start = time()
        self.cry_start = time()
        self.eye_close_start = time()
        self.eye_closed_at_angle_start = time()

    def start_rotation_at_degrees_per_second(self, degrees_per_second: int) -> None:
        self.rotating_image.start_rotation_at_degrees_per_second(degrees_per_second=degrees_per_second)

    def stop_rotation(self):
        self.rotating_image.stop_rotation()
        # self.rotation_velocity = 0

    @property
    def rotating(self):
        return self.rotating_image.rotating

    @property
    def original_image(self):
        return self.rotating_image.original_image

    def add_black_horizontal_line(self):
        p1, p2 = line_through_center_at_angle(
            centerx=self.radius,
            centery=self.radius,
            angle=0,
            length=self.radius * 2,  # np.pi/4,
        )
        cv2.line(self.img, p1, p2, (0, 0, 0), thickness=18) 

    def show_loading_animation(self, cycle_duration: float | None = None, direction: int | None = None):
        if cycle_duration is not None:
            self.loading_circle.set_cycle_duration(cycle_duration)
        if direction is not None:
            self.loading_circle.direction = direction
        self.loading_animation_active = True

    def synchronize_loading_animation(self, state_idx, last_step=None):
        self.loading_circle.synchronize_state(state_idx=state_idx, last_step=last_step)

    def loading_animation_state(self):
        return self.loading_circle.state_idx, self.loading_circle.last_step

    def stop_loading_animation(self):
        self.loading_animation_active = False
        self.img = self.original_image

    @property
    def color(self) -> tuple[int]:
        return self._color.bgr

    @property
    def smiling(self) -> bool:
        if self._smiling and (time() - self.smile_start) >= self.max_smiletime:
            self._smiling = False
        return self._smiling

    @property
    def crying(self) -> bool:
        if self._crying and (time() - self.cry_start) >= self.max_crytime:
            self._crying = False
        return self._crying

    @property
    def eye_closed(self) -> bool:
        if self._eye_closed and (time() - self.eye_close_start) >= self.max_eye_closetime:
            self._eye_closed = False
        return self._eye_closed

    @property
    def eye_closed_at_angle(self) -> bool:
        if self._eye_closed_at_angle and (time() - self.eye_closed_at_angle_start) >= self.max_eye_closetime:
            self._eye_closed_at_angle = False
        return self._eye_closed_at_angle

    @property
    def performing_gesture(self):
        return any([self.eye_closed, self.eye_closed_at_angle, self.crying, self.smiling])

    def start_crying(self, max_crytime: float | None = None) -> None:
        self._crying = True
        self.cry_start = time()
        if max_crytime is not None:
            self.max_crytime = max_crytime

    def start_smiling(self, max_smiletime: float | None = None) -> None:
        self._smiling = True
        self.smile_start = time()
        if max_smiletime is not None:
            self.max_smiletime = max_smiletime

    def close_eye(self, max_duration: float | None = None) -> None:
        self._eye_closed = True
        self.eye_close_start = time()
        if max_duration is not None:
            self.max_eye_closetime = max_duration

    def close_eye_at_angle(self, max_duration: float | None = None, angle: float | None = None) -> None:
        self._eye_closed_at_angle = True
        self.eye_closed_at_angle_start = time()
        if max_duration is not None:
            self.max_eye_closetime = max_duration
        if angle is not None:
            self.eye_closed_angle = math.radians(angle)

    def set_position_limits_in_eye(self, eye_width: int, eye_height: int) -> None:
        """Initializes position limits within an eye model"""
        self.min_x = self.radius
        self.max_x = eye_width - self.radius
        self.min_y = self.radius
        self.max_y = eye_height - self.radius
        self.eye_width = eye_width
        self.eye_height = eye_height
        self.update_position(eye_width // 2, eye_height // 2)  # initialize border coords

    def update_position(self, x: int, y: int) -> None:
        """Wrapper for border coordinate computation"""
        self.border_coords = self.center_position_to_border_coordinates(x, y)

    def center_position_to_border_coordinates(self, x: int, y: int) -> tuple[int, int, int, int]:
        """Compute iris edge coordinates from a known center position"""
        y1, y2 = int(y - self.radius), int(y + self.radius)
        x1, x2 = int(x - self.radius), int(x + self.radius)
        return self.correct_border_coordinates(y1, y2, x1, x2)

    def correct_border_coordinates(self, y1: int, y2: int, x1: int, x2: int) -> tuple[int, int, int, int]:
        """Ensure that boorder coordinates are within desired limits
        and stay the same relative to each other.

        set_position_limits_in_eye() must have been called
        once before using this to initialize eye_height and width correctly
        """
        if y1 <= 0:
            y2 = 1 + y2 - y1
            y1 = 1
        if y2 >= self.eye_height:
            y1 = self.eye_height - self.height - 1
            y2 = self.eye_height - 1
        if x1 <= 0:
            x2 = 1 + x2 - x1
            x1 = 1
        if x2 >= self.eye_width:
            x1 = self.eye_width - self.width - 1
            x2 = self.eye_width - 1
        return y1, y2, x1, x2

    def draw(self, target_img: cv2.typing.MatLike) -> None:
        """Puts iris on a given target_image"""
        y1, y2, x1, x2 = self.border_coords
        if self.visible:
            for c in range(3):
                target_img[y1:y2, x1:x2, c] = (
                    self.alpha_s * self.img[:, :, c] + self.alpha_l * target_img[y1:y2, x1:x2, c]
                )
        if self.smiling:
            cv2.rectangle(target_img, (x1, int(y1 + (y2 - y1) * 0.5)), (x2, y2), (0, 0, 0), -1)
        if self.crying:
            cv2.rectangle(target_img, (x1, y1), (x2, int(y1 + (y2 - y1) * 0.5)), (0, 0, 0), -1)
        if self.eye_closed:
            cv2.rectangle(target_img, (x1, y1), (x2, y2), (0, 0, 0), -1)
            draw_line_with_corners(
                target_img, (x1, int(y1 + (y2 - y1) * 0.5)), (x2, int(y1 + (y2 - y1) * 0.5)), self.color, thickness=24
            )
        if self.eye_closed_at_angle:
            cv2.rectangle(target_img, (x1, y1), (x2, y2), (0, 0, 0), -1)
            p1, p2 = line_through_center_at_angle(
                centerx=x2 - (x2 - x1) // 2,  # self.radius,
                centery=y2 - (y2 - y1) // 2,  # self.radius,
                angle=self.eye_closed_angle,
                length=self.radius * 2,  # np.pi/4,
            )
            draw_line_with_corners(target_img, p1, p2, self.color, thickness=24)

    def load_img(self, img_file: str, radius: int) -> None:
        """Iris image reading from file and resizing to a
        desired radius.

        Args:
        ----
          img_file (str): path to image file
        radius (int): target radius of image

        """
        iris = cv2.imread(str(img_file), -1)
        self.img = cv2.resize(iris, dsize=(int(radius * 2), int(radius * 2)))

    def build_img(self) -> None:
        """Iris image construction.
        Multiple styles are available.
        """
        if self.style == "simple":
            self.build_simple_img()
        elif self.style == "center_light":
            # first darken the current color
            color = tuple([int(0.9 * c) for c in self.color])
            # then build a simple iris
            self.build_simple_img(color=color)
            # and make the center brighter
            self.make_img_center_brighter(added_brightness=35)
        elif self.style == "beams":
            self.build_beams_img()
        else:
            self.build_complex_img()
        self.rotating_image.update_image(self.img)

    def build_simple_img(self, color: tuple[int, int, int, int] | None = None) -> None:
        """Flat 2D iris model with a
        primary color surrounded by a white border
        """
        if color is None:
            color = self.color
        canvas = np.ones((self.radius * 2, self.radius * 2, 4), dtype="uint8") * 255

        # white round canvas for a clean outer circle
        cv2.circle(
            canvas,
            (self.radius, self.radius),
            self.radius - 1,  # self.pupilradius,
            (255, 255, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        # circle of primary color at slightly reduced size
        cv2.circle(
            canvas,
            (self.radius, self.radius),
            self.radius - 6,  # self.pupilradius,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        self.img = canvas

    def change_color(self, color: tuple[int, int, int, int], duration: float = 0) -> None:
        self._color.update(*color, duration=duration)
        self.build_img()

    def update(self):
        if self._color.transitioning:
            self._color.transition_step()
            self.build_img()
        if self.rotating:
            self.img = self.rotating_image.img
        if self.loading_animation_active:
            self.img = cv2.subtract(self.original_image, self.loading_circle.img)

    def build_beams_img(self, color: tuple[int, int, int, int] | None = None) -> None:
        if color is None:
            color = self.color
        canvas = np.ones((self.radius * 2, self.radius * 2, 4), dtype="uint8") * 255
        # circle of primary color
        cv2.circle(
            canvas,
            (self.radius - 1, self.radius - 1),
            self.radius - 3,  # self.pupilradius,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        added_brightness = 20
        beamcolor = (color[0] + added_brightness, color[1] + added_brightness, color[2] + added_brightness)
        for angle in np.arange(0, np.pi * 2, np.pi / 32):  # [np.pi/4, np.pi/2, np.pi]:
            p1, p2 = line_through_center_at_angle(
                centerx=self.radius, centery=self.radius, angle=angle, length=self.radius * 2 - 10
            )
            cv2.line(canvas, p1, p2, beamcolor, thickness=2)

        self.img = canvas
        # self.start_rotation_at_velocity(velocity=1)
        # self.start_rotation_at_degrees_per_second(degrees_per_second=45)

    def make_img_center_brighter(self, added_brightness: int = 35) -> None:
        """Adds a bright spot to the inner iris"""
        logging.info("Center brightening")
        mask2 = np.zeros_like(self.img)
        cv2.circle(
            mask2,
            (self.radius, self.radius),
            int(self.radius * 0.7),
            (added_brightness,) * 3,
            thickness=-1,  # int(0.2*self.radius), #-1,
            lineType=cv2.LINE_AA,
        )
        # apply blur
        mask2 = cv2.GaussianBlur(mask2, (31, 31), int(self.radius * 0.7))
        alpha = 1 - mask2 / 255.0
        canvas = cv2.convertScaleAbs(self.img / alpha)
        # uncomment next line to make it darker instead:
        # canvas = cv2.convertScaleAbs(self.img*alpha)
        self.img = canvas

    def build_complex_img(self, ringdarkness: int = 180, blurdarkness: int = 150) -> None:
        """Iris with dark outer border

        Args:
        ----
            color (tuple): color in bgr (0-255 each)
            radius (int): radius of iris

        """
        radius = self.radius
        canvas = np.ones((radius * 2, radius * 2, 4), dtype="uint8") * 255

        # first draw a circle of the primary color
        cv2.circle(
            canvas,  # self.mirror.reflection,
            (radius, radius),
            radius,  # self.pupilradius,
            self.color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        # create a smooth dark border
        mask = np.zeros_like(canvas)
        cv2.circle(mask, (radius, radius), radius, (ringdarkness,) * 3, thickness=6, lineType=cv2.LINE_AA)  # -1,
        mask = cv2.GaussianBlur(mask, (5, 5), radius)
        alpha = 1 - mask / 255.0
        # blending
        canvas = cv2.convertScaleAbs(canvas * alpha)

        # add a gradually increasing darkening of outer iris
        mask2 = np.zeros_like(canvas)
        cv2.circle(
            mask2,
            (radius, radius),
            radius,
            (blurdarkness,) * 3,
            thickness=int(0.2 * radius),  # -1,
            lineType=cv2.LINE_AA,
        )
        # apply blur
        mask2 = cv2.GaussianBlur(mask2, (31, 31), radius)
        alpha = 1 - mask2 / 255.0
        canvas = cv2.convertScaleAbs(canvas * alpha)

        self.img = canvas
