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

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2

from mirror_eyes.custom_types import Extents, EyeForm, Limits, Offset, ScreenPoint
from mirror_eyes.tools.drawing import filter_nothing

IRIS_COLORS = {
    "yellow": (26, 224, 226),
    "lemon": (100, 214, 214),
    "blue": (255, 198, 154),
    "sky": (235, 206, 135),
    "pool": (253, 248, 186),
    "green": (145, 212, 93),
    "fgreen": (182, 220, 172),
    "forest": (76, 153, 0),
    "fblue": (227, 214, 184),
    "metalblue": (165, 136, 100),
    "red": (22, 31, 203),
    "lightred": (153, 153, 255),
    "brown": (0, 80, 153),
    "skin": (140, 180, 210),
    "darkbrown": (0, 49, 96),
    "white": (225, 225, 225),
    "purple": (255, 102, 178),
    "orange": (102, 178, 255),
    # "black": (0, 0, 0),
}


@dataclass
class DisplayConfig:
    size_pixel: Extents = Extents(width=1280, height=400)
    size_meters: Extents = Extents(width=0.24, height=0.0698)
    screen_position: ScreenPoint = ScreenPoint(x=80, y=100)
    screen_name: str = ""
    record_video: bool = False
    record_fps: int = 30
    show_fps: bool = False
    window_properties: tuple[int, ...] = (cv2.WND_PROP_TOPMOST, 1)
    window_flags: int = cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED #cv2.WINDOW_AUTOSIZE
    window_scale: int = 1.0  # specify by how much the window should be scaled relative to size_pixels


@dataclass
class PupilConfig:
    radius_range: Limits = Limits(min=60, max=140)
    radius_default: int = 110
    draw_behind_reflection: bool = True
    scale_to_face: bool = False
    variable_radius: bool = False
    follow_face: bool = True
    hysteresis: int = 1
    active: bool = True
    visible: bool = True
    color: tuple = (0, 0, 0, 0)
    whiteborder: bool = False
    blur_width: int = 10


@dataclass
class IrisConfig:
    color: tuple = IRIS_COLORS["white"]
    radius: int = 158  # 124
    active: bool = True
    visible: bool = True
    style: str = "pokeball"
    blur_width: int = 10


@dataclass
class CameraConfig:
    idx: int = 0
    resolution: Extents = Extents(width=1280, height=720)
    img_scale: float = 1.0  # 0.8


@dataclass
class OpacityConfig:
    opacity_range: Limits = Limits(min=1.0, max=1.0)
    fade_step: float = 0.02
    fadeout_duration: float = 5.0
    hysteresis: int = 14


@dataclass
class MirrorConfig:
    opacity: OpacityConfig = OpacityConfig()
    on_pupil: bool = True
    on_iris: bool = False
    active: bool = True
    moving: bool = True
    horizontal_flip: bool = True
    hysteresis: int = 10  # 12
    max_time: int = -1
    horizontal_offset: Offset = Offset(min=0.0, max=0.0, default=0.0, adaptive=False)
    vertical_offset: Offset = Offset(min=0.0, max=0.0, default=0.0, adaptive=True)
    size: Extents = Extents(width=320, height=320)
    camera: CameraConfig = CameraConfig()
    external_control: bool = True  # True for external eye_location control, False for automatic ROI+img-based control
    scale_to_roi: bool = True
    scale_hysteresis: int = 2
    filter:Callable = filter_nothing


@dataclass
class EyeConfig:
    pupil: PupilConfig = PupilConfig()
    iris: IrisConfig = IrisConfig()
    mirror: MirrorConfig = MirrorConfig()
    size: Extents = Extents(width=320, height=320)  # mirror.size
    hysteresis: int = 6
    screen_location: ScreenPoint = ScreenPoint(x=0, y=0)
    border_offset: int = 40
    form: EyeForm = EyeForm.EYE_PLUS_MIRROR_ONLY_ON_PUPIL
    record_video: bool = False
    recorder_fps: int = 30
    recording_directory: str = ""
    horizontal_offset: Offset = mirror.horizontal_offset
    name: str = "Eye"
    whitesclera: bool = False
    draw_frame: bool = False
    canvas_location: ScreenPoint = ScreenPoint(x=0, y=0)
    screen_x_range_pixel: Limits = Limits(min=0, max=DisplayConfig.size_pixel.width)
    distance_range_meters: Limits = Limits(min=0.15, max=1.5)
    ignore_all_input = False


@dataclass
class MirrorEyesConfig:
    name: str = "Mirror Eyes"
    eyes: list[EyeConfig] = field(default_factory=list)
    display: DisplayConfig = DisplayConfig()
    camera: CameraConfig = CameraConfig()