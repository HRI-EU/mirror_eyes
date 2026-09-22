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


from mirror_eyes.config.config_base import (
    IRIS_COLORS,
    DisplayConfig,
    EyeConfig,
    IrisConfig,
    MirrorConfig,
    MirrorEyesConfig,
    PupilConfig,
)
from mirror_eyes.custom_types import Extents, Limits, Offset, ScreenPoint
from mirror_eyes.tools.drawing import filter_nothing

# define configuration for local eye control
left_mirror_config = MirrorConfig(
    horizontal_flip=True,
    on_pupil=True,
    on_iris=True,
    external_control=False,
    scale_to_roi=True,
    horizontal_offset=Offset(min=-0.025, max=-0.1, default=-0.025, adaptive=True),
    size=Extents(width=400, height=400),
    filter = filter_nothing,
)
right_mirror_config = MirrorConfig(
    horizontal_flip=True,
    on_pupil=True,
    on_iris=True,
    external_control=False,
    scale_to_roi=True,
    horizontal_offset=Offset(min=0.025, max=0.1, default=0.025, adaptive=True),
    size=Extents(width=400, height=400),
    filter = filter_nothing
)

left_eye_config = EyeConfig(
    screen_location=ScreenPoint(x=200, y=0),
    mirror=left_mirror_config,
    name="Left Eye",
    pupil=PupilConfig(radius_default=80),
    iris=IrisConfig(style="eye", color=IRIS_COLORS['metalblue']),
    canvas_location=ScreenPoint(
        x=EyeConfig.border_offset - EyeConfig.size.width // 2, 
        y=int(DisplayConfig.size_pixel.height - left_mirror_config.size.height) // 2
    ),
    screen_x_range_pixel=Limits(min=int(DisplayConfig.size_pixel.width / 2), max=DisplayConfig.size_pixel.width),
    size=Extents(width=400, height=400),
)

right_eye_config = EyeConfig(
    screen_location=ScreenPoint(x=880, y=0),
    mirror=right_mirror_config,
    name="Right Eye",
    pupil=PupilConfig(radius_default=80),
    iris=IrisConfig(style="eye", color=IRIS_COLORS['metalblue']),
    canvas_location=ScreenPoint(
        x=int(DisplayConfig.size_pixel.width - EyeConfig.size.width // 2 - EyeConfig.border_offset),
        y=int(DisplayConfig.size_pixel.height - right_mirror_config.size.height) // 2,
    ),
    screen_x_range_pixel=Limits(min=0, max=int(DisplayConfig.size_pixel.width / 2)),
    size=Extents(width=400, height=400),
)

config_local_eye_control = MirrorEyesConfig(
    name="local_eye_control",
    eyes=[left_eye_config, right_eye_config],
    display=DisplayConfig(
        window_scale=1.,  # increase scale, e.g., for high dpi displays
    ),
    camera=left_eye_config.mirror.camera,
)