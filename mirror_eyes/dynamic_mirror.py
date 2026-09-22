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

# external
import cv2
import numpy as np

# own
import mirror_eyes.config.config as cfg
from mirror_eyes.custom_types import ScreenPoint
from mirror_eyes.tools.image_scaling import add_blended_margin


class DynamicMirror:
    """Class that handles mirror image manipulation for an Eye model. """
    def __init__(self, mirror_cfg: cfg.MirrorConfig | None = None) -> None:
        self.cfg = mirror_cfg
        self.horizontal_offset = mirror_cfg.horizontal_offset.default 
        self.vertical_offset = mirror_cfg.vertical_offset
        self._normalized_pos = np.array([0.5, 0.5], dtype=np.float32)

        self.smoothing_factor = 1.0 / mirror_cfg.hysteresis if mirror_cfg.hysteresis > 0 else 1.0 

        self.opacity = mirror_cfg.opacity.opacity_range.min
        if mirror_cfg.opacity.hysteresis > 0:
            self.opacity_smoothing_factor = 1.0 / mirror_cfg.opacity.hysteresis
        else:
            self.opacity_smoothing_factor = 1.0

        self.moving_image = mirror_cfg.moving
        self.horizontal_flip = mirror_cfg.horizontal_flip
        self.img_scale = 1.0
        self.img_scale_based_offset = ScreenPoint(x=0, y=0)

        # Use np arrays for dimensions to enable vectorized arithmetic
        self.box_size = np.array(
            [mirror_cfg.size.width, mirror_cfg.size.height], 
            dtype=np.int32)
        self.max_res = np.array(
            [mirror_cfg.camera.resolution.width, mirror_cfg.camera.resolution.height], 
            dtype=np.int32)
        
        # Pre-calculate the extended margin
        self.extended_image_margin = int(mirror_cfg.size.width / 2)
        self.max_res_extended = self.max_res + (self.extended_image_margin * 2)
        
        # Current state of the crop (top_left_x, top_left_y)
        self.top_left = np.array([0, 0], dtype=np.int32)
        self.eye_pos = ScreenPoint(x=0, y=0) # np.array([0, 0], dtype=np.int32)

        self.reflection = None
        self.reflection_filter = mirror_cfg.filter #filter_nothing
        logging.info("Initialized dynamic mirror")

    @property
    def x(self):
        return self.eye_pos.x

    @x.setter
    def x(self, value):
        self.eye_pos.x = value

    @property
    def y(self):
        return self.eye_pos.y 

    @y.setter
    def y(self, value):
        self.eye_pos.y = value

    # properties with integrated smoothing
    @property
    def height(self):
        return self.box_size[1]

    @height.setter
    def height(self, value: int):
        self.box_size[1] = value
        # Trigger boundary correction to ensure the box still fits in max_res
        self.update_image_cropping_area()

    @property
    def width(self):
        return self.box_size[0]

    @width.setter
    def width(self, value: int):
        self.box_size[0] = value
        self.update_image_cropping_area()

    @property
    def normalized_x(self):
        #return self._normalized_x
        return self._normalized_pos[0]

    @normalized_x.setter
    def normalized_x(self, value: float):
        self.normalized_x = (value * self.smoothing_factor) + \
            (self.normalized_x * (1 - self.smoothing_factor))

    @property
    def normalized_y(self):
        return self._normalized_pos[1]

    @normalized_y.setter
    def normalized_y(self, value: float):
        self.normalized_y = (value * self.smoothing_factor) + \
            (self.normalized_y * (1 - self.smoothing_factor))

    def update_opacity(self, opacity: float) -> None:
        """Write opacity into a deque for automatic fading"""
        #self.opacity_history.append(opacity)
        #self.opacity = np.mean(self.opacity_history)
        self.opacity = (
            opacity * self.opacity_smoothing_factor) + \
                (self.opacity * (1.0 - self.opacity_smoothing_factor))

    def update_camera_resolution(self, width: int, height: int) -> None:
        logging.info(f"Updating camera resolution to {width}x{height}")
        self.max_res = np.array([width, height], dtype=np.int32)
        self.max_res_extended = self.max_res + (self.extended_image_margin * 2)

    def update_img_scale(self, img_scale: float = 1.0, img_scale_based_offset: ScreenPoint | None = None) -> None:
        self.img_scale = min(max(img_scale, 0.0), 1.0)
        if img_scale_based_offset is not None:
            self.img_scale_based_offset = img_scale_based_offset

    def normalize_full_image_poi(self, full_image_x, full_image_y) -> None:
        new_pos = np.array([full_image_x, full_image_y]) / self.max_res
        self._normalized_pos = (
            new_pos * self.smoothing_factor) + \
                (self._normalized_pos * (1.0 - self.smoothing_factor))

    def update_image_cropping_area(self) -> None:
        avail = (self.img_scale * self.max_res) - self.box_size

        # top_left = offset + [avail_w * (1-norm_x), avail_h * norm_y]
        multiplier = np.array(
            [1.0 - self._normalized_pos[0], 
            self._normalized_pos[1]], dtype=np.float32)
        top_left = self.img_scale_based_offset + (avail * multiplier)

        # boundary correction with np.clip instead of bbox 
        limit_low = np.array([0, 0])
        limit_high = self.max_res - self.box_size 
        self.top_left = np.clip(top_left, limit_low, limit_high).astype(np.int32)

    def create_reflection_crop(self, image: cv2.typing.MatLike) -> None:
        """Crops a given input image to eye size
        and flips it horizontally. Assumes the input image has dimensions self.cam_width * self.cam_height
        """
        try: 
            # margin expansion
            extended_image = add_blended_margin(image=image, margin=self.extended_image_margin)

            # calculate extended crop coordinates 
            ext_tl = self.top_left + self.extended_image_margin

            # slicing 
            nimage = extended_image[
                ext_tl[1] : ext_tl[1] + self.height,
                ext_tl[0] : ext_tl[0] + self.width,
            ]

            if self.horizontal_flip:
                nimage = cv2.flip(nimage, 1)
            
            self.reflection = self.reflection_filter(nimage)

        except Exception as e:
            logging.error(f"Reflection crop generation failed. Not updating reflection image. {e}")

    def convert_camera_coordinates_to_eye_coordinates(self, x: int, y: int) -> None:
        """Convert x,y from camera to eye/mirror coordinate space"""
        point = np.array([x, y], dtype=np.float32)
        norm = point / self.max_res
        upper_limit = self.max_res - self.box_size
        self.eye_pos = ScreenPoint.from_array((point - (norm * upper_limit)).astype(np.int32))

    def update_all(
        self,
        x_camera: int,
        y_camera: int,
        opacity: float = 1.0,
        x_on_eye: int | None = None,
        y_on_eye: int | None = None,
        horizontal_offset: int = 0,
    ):
        """Bundles updates for horizontal_offset,
        opacity, image cropping and coordinate conversion
        """
        self.horizontal_offset = horizontal_offset
        self.update_opacity(opacity)
        if self.moving_image:
            self.normalize_full_image_poi(full_image_x=x_camera, full_image_y=y_camera)
            self.update_image_cropping_area()
        if x_on_eye is not None and y_on_eye is not None:
            #self.x = x_on_eye
            #self.y = y_on_eye
            self.eye_pos = ScreenPoint(x=x_on_eye, y=y_on_eye)
        else:
            self.convert_camera_coordinates_to_eye_coordinates(x_camera, y_camera)


class DynamicMirrorExtCtrl(DynamicMirror):
    """Variant of dynamic mirror for which image crop
    and placement locations are controlled externally"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def update_image_cropping_area(self) -> None:
        # total scaled area available
        scaled_max = self.img_scale * self.max_res

        # center point from normalized position
        target_center = self.img_scale_based_offset + (scaled_max * self._normalized_pos)

        # shift to topleft corner for bounding box
        top_left = target_center - self.box_size/2

        # clip to boundaries
        limit_low = np.array([0, 0])
        limit_high = self.max_res - self.box_size
        self.top_left = np.clip(top_left, limit_low, limit_high).astype(np.int32)

