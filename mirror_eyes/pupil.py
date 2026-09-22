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

import cv2

# own
import mirror_eyes.config.config as cfg
from mirror_eyes.tools.converting import restrict_to_range
from mirror_eyes.tools.drawing import make_blur_circle


class Pupil:
    """Pupil object used by EyeModel.
    Contains functionality for plotting and
    pupil size adaptations.
    """

    def __init__(self, pupil_cfg: cfg.PupilConfig = None) -> None:
        """Args:
        ----
          pupil_cfg (PupilConfig): configuration class containing details about radius, color, border,
          visibility, etc.

        """
        self.cfg = pupil_cfg
        self.color = self.cfg.color
        self.whiteborder = self.cfg.whiteborder
        self._radius = self.cfg.radius_default
        self.radius_min = self.cfg.radius_range.min
        self.radius_max = self.cfg.radius_range.max
        self.variable_radius = self.cfg.variable_radius
        self.hysteresis = self.cfg.hysteresis
        self.radius_smoothing = self.hysteresis > 1
        self.smoothing_factor = 1.0 / self.hysteresis if self.hysteresis > 0 else 1.0 

        self.visible = self.cfg.visible

        self.blur_circle = make_blur_circle(radius=self._radius, blurwidth=self.cfg.blur_width)
        self.set_position_limits_in_eye(self._radius * 4, self._radius * 4, self._radius * 2)
        logging.info(f"Pupil initialized with size {self._radius} and color {self.color}")

    @property
    def radius(self) -> int:
        """Get radius"""
        return self._radius

    @radius.setter
    def radius(self, value: int) -> None:
        """Set radius with optional smoothing"""
        value = restrict_to_range(value, lower=self.radius_min, upper=self.radius_max)

        if self.radius_smoothing:
            self._radius = int((value * self.smoothing_factor) + (self._radius * (1 - self.smoothing_factor)))
        else:
            self._radius = value

    def set_position_limits_in_eye(
        self, eye_width: int, eye_height: int, iris_radius: int, displacement_lim=0.9
    ) -> None:
        """Initializes position limits within an eye model"""
        self.min_x = int(iris_radius * displacement_lim)  # used for 3d pupil effect
        self.max_x = eye_width - self.min_x
        self.min_y = int(iris_radius * displacement_lim)
        self.max_y = eye_height - self.min_y

    def distance_to_radius(self, distance: float) -> None:
        """Adapt the pupil radius to the distance to the point of interest

        Args:
        ----
            distance (int): normalized distance to gaze object

        """
        try:
            radius = restrict_to_range(
                self.radius_min + (1 - distance - 0.3) * (self.radius_max - self.radius_min),
                lower=self.radius_min,
                upper=self.radius_max,
            )
        except NameError as err:
            logging.error(err)
            radius = self.radius
        self.radius = radius

    def update(self, distance: float) -> None:
        """Updates pupil radius based on a distance value"""
        if self.variable_radius:
            self.distance_to_radius(distance)

    def draw(self, img: cv2.typing.MatLike, x: int, y: int) -> None:
        """Draws pupil on an image at a given position."""
        if self.visible:
            cv2.circle(img, (x, y), self.radius, self.color, thickness=-1, lineType=cv2.LINE_AA)
            if self.whiteborder:
                # add a white ring
                cv2.circle(img, (x, y), self.radius, (250, 250, 250), thickness=3, lineType=cv2.LINE_AA)

    def load_img(self, img_file: str) -> None:
        """Pupil image reading from file and resizing to a
        desired radius.

        Args:
        ----
          img_file (str): path to image file
        """
        img = cv2.imread(str(img_file), -1)
        img = cv2.resize(img, dsize=(int(self.radius * 2), int(self.radius * 2)))
        return img
