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

import cv2

from mirror_eyes.config.config_base import (
    DisplayConfig,
    EyeConfig,
    IrisConfig,
    MirrorConfig,
    MirrorEyesConfig,
)
from mirror_eyes.custom_types import Limits, Offset, ScreenPoint

# define configuration for external eye control
head_display_config = DisplayConfig(
    screen_position=ScreenPoint(x=80, y=100),  # ScreenPoint(x=1920, y=0), # switch back after testing
    window_properties=(cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN),
)

left_mirror_config_external = MirrorConfig(
    horizontal_flip=True,
    external_control=True,
    horizontal_offset=Offset(min=-0.025, max=-0.1, default=-0.025, adaptive=True),
    scale_to_roi=True,
)
right_mirror_config_external = MirrorConfig(
    horizontal_flip=True,
    external_control=True,
    horizontal_offset=Offset(min=0.025, max=0.1, default=0.025, adaptive=True),
    scale_to_roi=True,
)

left_eye_config_external = EyeConfig(
    screen_location=ScreenPoint(x=200, y=0),
    mirror=left_mirror_config_external,
    name="Left Eye",
    iris=IrisConfig(style="pokeball"),
    canvas_location=ScreenPoint(
        x=EyeConfig.border_offset, y=int(DisplayConfig.size_pixel.height - left_mirror_config_external.size.height) // 2
    ),
    screen_x_range_pixel=Limits(min=int(DisplayConfig.size_pixel.width / 2), max=DisplayConfig.size_pixel.width),
)

right_eye_config_external = EyeConfig(
    screen_location=ScreenPoint(x=880, y=0),
    mirror=right_mirror_config_external,
    name="Right Eye",
    iris=IrisConfig(style="pokeball"),
    canvas_location=ScreenPoint(
        x=int(DisplayConfig.size_pixel.width - EyeConfig.size.width - EyeConfig.border_offset),
        y=int(DisplayConfig.size_pixel.height - right_mirror_config_external.size.height) // 2,
    ),
    screen_x_range_pixel=Limits(min=0, max=int(DisplayConfig.size_pixel.width / 2)),
)


config_external_eye_control = MirrorEyesConfig(
    name="external_eye_control", eyes=[left_eye_config_external, right_eye_config_external], display=head_display_config
)