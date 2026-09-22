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

import random
from collections import deque
from time import time

import numpy as np

from mirror_eyes.custom_types import Extents, Limits, ScreenPoint


class BoundingBox2D:
    """Bounding box defined by top left corner point and extents (with, height).
    Provides access to other points of interest (other corners and center).
    Applies boundary corrections if max_extents are specified"""

    def __init__(
        self, top_left: ScreenPoint, extents: Extents, max_extents: Extents | None = None, padding: float | None = None
    ) -> None:
        if padding is not None:
            top_left, extents = self.add_padding(top_left=top_left, extents=extents, padding=padding)
        self._extents = extents
        self._top_left = top_left
        self.max_extents = max_extents
        self.keep_box_within_bounds = max_extents is not None

        if self.keep_box_within_bounds and not self.bounding_box_within_valid_range():
            # Automatically correct invalid bounding boxes
            self._top_left, self._extents = self.boundary_correction(self._top_left, self._extents)

    @classmethod
    def from_points(
        cls, 
        points: np.ndarray, 
        padding: float | None = None, 
        max_extents: Extents | None = None) -> BoundingBox2D:
        """Alternative initialization from an
        array of 2D points based on location
        minima and maxima.
        """
        xy_min = np.min(points, axis=0)
        xy_max = np.max(points, axis=0)
        top_left = ScreenPoint(x=xy_min[0], y=xy_min[1])
        extents = Extents(width=xy_max[0] - xy_min[0], height=xy_max[1] - xy_min[1])
        return cls(top_left=top_left, extents=extents, max_extents=max_extents, padding=padding)

    def __str__(self):
        return f"2D Bounding box at {self.top_left} with extents {self.extents}"

    @property
    def top_left(self) -> ScreenPoint:
        return self._top_left

    @top_left.setter
    def top_left(self, value: ScreenPoint):
        self._top_left = value

    @property
    def extents(self) -> Extents:
        return self._extents

    @extents.setter
    def extents(self, value: Extents):
        self._extents = value

    @property
    def width(self) -> int:
        return self.extents.width

    @property
    def height(self) -> int:
        return self.extents.height

    @property
    def top_right(self) -> ScreenPoint:
        return ScreenPoint(x=self.top_left.x + self.width, y=self.top_left.y)

    @property
    def bottom_left(self) -> ScreenPoint:
        return ScreenPoint(x=self.top_left.x, y=self.top_left.y + self.height)

    @property
    def bottom_right(self) -> ScreenPoint:
        return ScreenPoint(x=self.top_left.x + self.width, y=self.top_left.y + self.height)

    @property
    def center(self) -> ScreenPoint:
        return ScreenPoint(x=self.top_left.x + self.width // 2, y=self.top_left.y + self.height // 2)

    @property
    def top_left_upper_limit(self) -> ScreenPoint:
        if not self.max_extents:
            raise ValueError("max_extents must be defined to calculate upper limit")
        return ScreenPoint(x=self.max_extents.width - self.width, y=self.max_extents.height - self.height)

    @property
    def left_margin(self) -> int:
        return self.top_left.x

    @property
    def top_margin(self) -> int:
        return self.top_left.y

    @property
    def right_margin(self) -> int:
        return max(0, self.max_extents.width - self.bottom_right.x)

    @property
    def bottom_margin(self) -> int:
        return max(0, self.max_extents.height - self.bottom_right.y)

    def inner_box(self, relative_size: float = 0.3) -> BoundingBox2D:
        if relative_size >= 1:
            return self
        new_w = int(self.width * relative_size)
        new_h = int(self.height * relative_size)
        return BoundingBox2D(
            top_left=ScreenPoint(
                x=self.center.x - new_w // 2,
                y=self.center.y - new_h // 2,
            ),
            extents=Extents(width=new_w, height=new_h),
        )

    def point_from_id(self, pid: int) -> ScreenPoint:
        mapping = {
            0: self.center,
            1: self.top_left, 
            2: self.top_right,
            3: self.bottom_left,
            4: self.bottom_right
        }
        if pid in mapping:
            return mapping[pid]
        return self.inner_box(relative_size=0.5).point_from_id(pid - 4)

    def add_padding(self, top_left: ScreenPoint, extents: Extents, padding: float) -> tuple[ScreenPoint, Extents]:
        padding_pixel = int(extents.width * padding)
        half_padding = padding_pixel // 2
        new_top_left = ScreenPoint(x=top_left.x - half_padding, y=top_left.y - half_padding)
        new_extents = Extents(width=extents.width + padding_pixel, height=extents.height + padding_pixel)
        return new_top_left, new_extents

    def extents_within_valid_range(self) -> bool:
        if self.max_extents is None:
            return True
        return (
            0.0 < self.extents.height <= self.max_extents.height \
                and 0.0 < self.extents.width <= self.max_extents.width
        )

    def point_within_valid_range(self, point: ScreenPoint) -> bool:
        if self.max_extents is None:
            return True
        return 0 <= point.y <= self.max_extents.height and 0 <= point.x <= self.max_extents.width

    def bounding_box_within_valid_range(self) -> bool:
        return self.point_within_valid_range(self.top_left) and self.point_within_valid_range(self.bottom_right)

    def boundary_correction(self, top_left: ScreenPoint, extents: Extents) -> tuple[ScreenPoint, Extents]:

        tl = ScreenPoint(top_left.x, top_left.y)
        ex = Extents(extents.width, extents.height)

        tl.x = max(0, tl.x)
        tl.y = max(0, tl.y)

        if self.max_extents:
            ex.width = min(ex.width, self.max_extents.width)
            ex.height = min(ex.height, self.max_extents.height)

            # push back into screen if it overshoots right or bottom 
            if (tl.x + ex.width) > self.max_extents.width:
                tl.x -= (tl.x + ex.width) - self.max_extents.width 
            if (tl.y + ex.height) > self.max_extents.height:
                tl.y -= (tl.y + ex.height) - self.max_extents.height
        return tl, ex 

    def update(
        self,
        top_left: ScreenPoint | None = None,
        center: ScreenPoint | None = None,
        extents: Extents | None = None,
        boundary_correction: bool = True,
        padding: float | None = None,
    ) -> None:
        if padding is not None:
            top_left, extents = self.add_padding(
                top_left=top_left, extents=extents, padding=padding
            )
        if top_left is None:
            top_left = self.top_left
        if extents is None:
            extents = self.extents

        if padding is not None:
            top_left, extents = self.add_padding(top_left, extents, padding)
        
        if center is not None:
            top_left = ScreenPoint(
                x=center.x - extents.width // 2,
                y=center.y - extents.height // 2,
            )
        if self.keep_box_within_bounds or boundary_correction:
            top_left, extents = self.boundary_correction(
                top_left=top_left, extents=extents
            )
        self.top_left = top_left
        self.extents = extents


class SmoothBoundingBox2D(BoundingBox2D):
    """Bounding box that applies temporal smoothing on corner and extents samples.
    """

    def __init__(
        self,
        top_left: ScreenPoint,
        extents: Extents,
        hysteresis: int = 6,
        max_extents: Extents | None = None,
        padding: float | None = None,
    ) -> None:
        # initialize base class to handle padding and boundaries 
        super().__init__(top_left=top_left, extents=extents, max_extents=max_extents, padding=padding)
        self.smoothing_factor = 1.0 / hysteresis if hysteresis > 0 else 1.0 
        self.offset = ScreenPoint(x=0, y=0)

    @property
    def top_left(self) -> ScreenPoint:
        return self._top_left

    @top_left.setter
    def top_left(self, new_point: ScreenPoint) -> None:
        self._top_left = ScreenPoint.from_array(
            (new_point * self.smoothing_factor) + \
                (self._top_left * (1.0 - self.smoothing_factor))) + self.offset

    @property
    def extents(self) -> Extents:
        return self._extents

    @extents.setter
    def extents(self, new_extents: Extents) -> None:
        self._extents = Extents.from_array((new_extents * self.smoothing_factor) + \
            (self._extents * (1.0 - self.smoothing_factor)))

    def apply_offset(self, offset: ScreenPoint):
        self.offset = offset 


class RoiScanIndexer:
    """Support object to generate sequences of gaze point indices
    for simulated scanning events. The length of such an event can
    be specified.
    Individual index focus durations vary randomly between specified
    limits.
    The focus point indices should be compatible with the indices of
    a BoundingBox instance. For instance, a bounding box with
    four corners has five indices, 1-4 for the corners and
    0 for the center."""

    def __init__(
        self,
        index_count: int = 5,
        max_roi_scan_duration: float = 2.0,
        focus_point_duration_range: Limits = Limits(min=0.2, max=1.5),
    ):
        self.focus_point_indices = deque(list(range(index_count)), maxlen=index_count)
        self.max_roi_scan_duration = max_roi_scan_duration  # seconds
        self.focus_point_duration = focus_point_duration_range.min  # seconds
        self.focus_point_duration_range = focus_point_duration_range
        self.scan_start = time()
        self.focus_point_start = time()
        self._active = False

    def start(self, max_duration: float | None = None, focus_point_duration_range: Limits | None = None) -> None:
        random.shuffle(self.focus_point_indices)
        self._active = True
        self.scan_start = time()
        self.focus_point_start = time()
        if max_duration is not None:
            self.max_roi_scan_duration = max_duration
        if focus_point_duration_range is not None:
            self.focus_point_duration_range = focus_point_duration_range

    @property
    def active(self):
        if self._active and (time() - self.scan_start) >= self.max_roi_scan_duration:
            self._active = False
        elif self._active and (time() - self.focus_point_start) >= self.focus_point_duration:
            self.advance_focus_point()
        return self._active

    def advance_focus_point(self) -> None:
        # rotate the index queue
        self.focus_point_indices.append(self.focus_point_indices[0])
        self.focus_point_start = time()
        self.focus_point_duration = max(
            self.focus_point_duration_range.min, random.random() * self.focus_point_duration_range.max
        )
