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

# pylint: disable=E1101,E0401,C0415,logging-fstring-interpolation

# standard
from __future__ import annotations

import logging
from pathlib import Path

# external
import cv2
import numpy as np

# own
import mirror_eyes.config.config as cfg
from mirror_eyes.dynamic_mirror import DynamicMirror, DynamicMirrorExtCtrl
from mirror_eyes.iris import Iris
from mirror_eyes.pupil import Pupil
from mirror_eyes.tools.converting import apply_noise, euclidean_distance, normalize, restrict_to_range
from mirror_eyes.tools.drawing import apply_blur_circle_mask
from mirror_eyes.tools.video_recording import VideoRecorder


class EyeModel:
    """Model for a single eye.
    Unifies functionality for drawing an eye model
    according to various specifications and for displaying
    the behavior of that eye model.
    It carries out some input smoothing to improve appearance.
    It takes care of camera image cropping for mirroring
    with background movement and can apply horizontal and vertical
    offsets to simulate ocular disparity.
    """

    def __init__(self, eye_cfg: cfg.EyeConfig, framefn: str = "frame.png", mediadir: str | None = None) -> None:
        """Args:
        ----
          eye_cfg (EyeConfig) - full configuration specification
          framefn (str) - filename of frame image
          mediadir (str) - directory containing framefn
        """
        self.cfg = eye_cfg

        self.pupil = Pupil(pupil_cfg=self.cfg.pupil)
        self.iris = Iris(iris_cfg=self.cfg.iris)
        if self.cfg.mirror.external_control:
            self.mirror = DynamicMirrorExtCtrl(mirror_cfg=self.cfg.mirror)
        else:
            self.mirror = DynamicMirror(mirror_cfg=self.cfg.mirror)

        self.screen_location = self.cfg.screen_location
        self.canvas_location = self.cfg.canvas_location  
        self.show_mirror = self.cfg.mirror.active
        self.show_iris = self.cfg.iris.active
        self.show_pupil = self.cfg.pupil.active
        self.mirror_on_pupil = self.cfg.mirror.on_pupil
        self.mirror_on_iris = self.cfg.mirror.on_iris
        self.draw_pupil_behind_reflection = self.cfg.pupil.draw_behind_reflection
        self.whitesclera = self.cfg.whitesclera
        self.name = self.cfg.name
        
        self.smoothing_factor = 1.0 / self.cfg.hysteresis if self.cfg.hysteresis > 0 else 1.0 

        self.pupil_follow_target = self.cfg.pupil.follow_face
        self.horizontal_offset = self.cfg.horizontal_offset.default
        self.min_horizontal_offset = self.cfg.horizontal_offset.min
        self.max_horizontal_offset = self.cfg.horizontal_offset.max
        self.adaptive_horizontal_offset = self.cfg.horizontal_offset.adaptive
        self.vertical_offset = 0
        self.height = self.cfg.size.height
        self.width = self.cfg.size.width
        record_video = self.cfg.record_video
        video_fps = self.cfg.recorder_fps
        framefn = None
        draw_frame = self.cfg.draw_frame

        if mediadir is None:
            mediadir = Path(__file__).parent / "media"

        if framefn is None:
            self.build_no_frame(width=self.width, height=self.height)
        else:
            framepath = mediadir / framefn
            self.frame = cv2.imread(str(framepath.resolve()), -1)
        if draw_frame:
            self.build_frame()

        if not self.whitesclera:
            self.sclera = np.zeros_like(self.frame[:, :, :3], dtype="uint8")
        else:
            self.sclera = np.ones_like(self.frame[:, :, :3], dtype="uint8") * 255
            self.draw_pupil_behind_reflection = True
        # self.refresh_outimg()
        self.outimg = self.sclera.copy()

        # self.height, self.width = self.frame.shape[:2]
        # update mirror settings
        self.mirror.width = self.width
        self.mirror.height = self.height
        self._x, self._y = int(self.width / 2), int(self.height / 2)

        self.pupil.set_position_limits_in_eye(
            eye_width=self.width, eye_height=self.height, iris_radius=self.iris.radius
        )
        self.iris.set_position_limits_in_eye(eye_width=self.width, eye_height=self.height)
        # optional recording of individual eye videos
        self.video_recorder = VideoRecorder(
            height=self.height,
            width=self.width,
            fps=video_fps,
            name=self.name,
            active=record_video,
            directory=self.cfg.recording_directory,
        )
        logging.info(f"Eye model size: {self.width}x{self.height} px")
        logging.info(f'Initalized eye model "{self.name}"')

    @property
    def x(self) -> int:
        """Get x coordinate"""
        return self._x

    @x.setter
    def x(self, value: int) -> None:
        """Setter with coordinate constraints"""
        # the following additions lead to a gradual movement of the pupil
        # relative to the iris at the eye borders
        if value < self.iris.min_x:
            x = int(
                normalize(
                    value,  # +int(self.horizontal_offset*self.width),
                    min_source=0,
                    max_source=self.iris.radius,  # self.width,
                    min_target=self.pupil.min_x,
                    max_target=self.iris.radius,  # self.pupil_max_x
                )
            )
        elif value > self.iris.max_x:
            x = int(
                normalize(
                    value,
                    min_source=self.iris.max_x,
                    max_source=self.width,
                    min_target=self.iris.max_x,
                    max_target=self.pupil.max_x,
                )
            )
        else:
            x = value
        # smoothing
        self.prev_x = self._x 
        self._x = int((x * self.smoothing_factor) + (self._x * (1.0 - self.smoothing_factor)))

    @property
    def y(self) -> int:
        """Get y coordinate"""
        return self._y

    @y.setter
    def y(self, value: int) -> None:
        """Setter with coordinate constraints"""
        if value < self.iris.min_y:
            y = int(
                normalize(
                    value,  # +int(self.vertical_offset*self.height),
                    min_source=0,
                    max_source=self.iris.radius,  # self.height,
                    min_target=self.pupil.min_y,
                    max_target=self.iris.radius,  # self.pupil_max_y
                )
            )
        elif value > self.iris.max_y:
            y = int(
                normalize(
                    value,
                    min_source=self.iris.max_y,
                    max_source=self.height,
                    min_target=self.iris.max_y,
                    max_target=self.pupil.max_y,
                )
            )
        else:
            y = value
        # apply smoothing
        self.prev_y = self._y
        self._y = int((y * self.smoothing_factor) + (self._y * (1.0 - self.smoothing_factor)))


    @property
    def saccade(self):
        """simplistic saccade classification
        if the distance between the current gaze point and the recent average is above threshold_distance, the current
        state is classified as a saccade.

        Note:
        A high-level approach can be preferable
        e.g., driven by the attention module.
        With the present low-level approach there is the risk of
        classifying fast smooth pursuit as a saccade."""
        threshold_distance = 50  # <- can be optimized
        distance = euclidean_distance((self.prev_x, self.prev_y), (self._x, self._y))
        return distance > threshold_distance

    @property
    def mirror_disabled_by_action(self) -> bool:
        return any(
            [
                self.iris.eye_closed,
                self.iris.eye_closed_at_angle,
                self.iris.crying,
                self.iris.smiling,
                self.cfg.ignore_all_input,
                # self.saccade,  # emulating animal model: No perception is taking place during saccades
            ]
        )

    @property
    def pupil_disabled_by_action(self) -> bool:
        return any([self.iris.eye_closed, self.iris.eye_closed_at_angle])

    def change_eye_color(self, color: tuple[int, int, int, int], duration: float = 0.0) -> None:
        if not self.cfg.ignore_all_input:
            self.iris.change_color(color=color, duration=duration)

    def rotate_iris_at_degrees_per_second(self, degrees_per_second: int) -> None:
        if not self.cfg.ignore_all_input:
            self.iris.start_rotation_at_degrees_per_second(degrees_per_second=degrees_per_second)

    def stop_iris_rotation(self) -> None:
        self.iris.stop_rotation()

    def show_loading_animation(self, cycle_duration: float | None = None, direction: int | None = None) -> None:
        if not self.cfg.ignore_all_input:
            self.iris.show_loading_animation(cycle_duration=cycle_duration, direction=direction)

    def stop_loading_animation(self) -> None:
        self.iris.stop_loading_animation()

    def loading_animation_state(self) -> None:
        return self.iris.loading_animation_state()

    def synchronize_loading_animation(self, state_idx, last_step=None) -> None:
        self.iris.synchronize_loading_animation(state_idx=state_idx, last_step=last_step)

    def update_img_scale(self, img_scale=1.0, img_scale_based_offset=None) -> None:
        self.mirror.update_img_scale(img_scale=img_scale, img_scale_based_offset=img_scale_based_offset)

    def update_camera_resolution(self, width: int, height: int) -> None:
        self.mirror.update_camera_resolution(width=width, height=height)

    def shutdown(self) -> None:
        """Shutdown procedure"""
        self.video_recorder.shutdown()
        logging.info(f"{self.name} shutdown complete")

    def enable_mirroring(self) -> None:
        self.show_mirror = True

    def disable_mirroring(self) -> None:
        self.show_mirror = False

    def update_position(self, x: int, y: int, microsaccades: bool = False) -> None:
        """Append input location to history and determine current
        x and y coordinates as the average of their recent
        history for smoothing.
        """
        if microsaccades:
            self.x = apply_noise(x)
            self.y = apply_noise(y)
        else:
            self.x = x
            self.y = y
        self.iris.update_position(self.x, self.y)

    def x_to_horizontal_offset(self, x: float, min_x: float = 0.3, max_x: float = 0.8) -> None:
        """Simulate adaptive binocular disparity in relation to focus distance
        Focus distance is approximated by x
            x can be the relative eye-space occupied by a tracked face.
        Assumption: the larger x the closer the face, the more
        center focus is required. The default values for min_x and
        max_x are selected with that assumption
            x can also be an actual distance measure or estimate.
        In that case appropriate min_x and max_x distance numbers
        should be provided for horizontal offset normalization
        within a suitable distance range.

        Args:
        ----
            x (float): focus object size relative to the eye
            min_x (float): x at and below which min_offset is applied
            max_x (float): x at and above which max_offset is applied

        """
        self.horizontal_offset = restrict_to_range(
            normalize(
                val_source=restrict_to_range(x, lower=min_x, upper=max_x),
                min_source=min_x,
                max_source=max_x,
                min_target=self.min_horizontal_offset,  # minimum offset as fraction of eye width
                max_target=self.max_horizontal_offset,
            ),
            self.min_horizontal_offset,
            self.max_horizontal_offset,
        )
        # uncomment to also update horizontal offset for mirror
        # self.mirror.horizontal_offset = self.horizontal_offset

    def update_model(
        self,
        camera_image: cv2.typing.MatLike,
        x_camera: int,
        y_camera: int,
        distance: float,
        opacity: float,
        x_on_eye: int | None = None,
        y_on_eye: int | None = None,
    ) -> None:
        """Model updates"""
        # if crop_bounding_box is None:
        # self.mirror.create_reflection_crop(camera_image)
        # else:
        #    self.mirror.crop_camera_image_and_flip(camera_image, crop_bounding_box=crop_bounding_box)
        if self.adaptive_horizontal_offset:
            self.x_to_horizontal_offset(
                x=distance,
                min_x=1,  # 500, # uncomment for unnormalized distance
                max_x=0.2,  # 0, #200,
            )
        self.pupil.update(distance)
        self.iris.update()
        if not self.cfg.ignore_all_input:
            self.mirror.update_all(
                x_camera=x_camera, y_camera=y_camera, opacity=opacity, x_on_eye=x_on_eye, y_on_eye=y_on_eye
            )
            self.mirror.create_reflection_crop(camera_image)
            if self.pupil_follow_target:
                self.update_position(self.mirror.x, self.mirror.y)
            else:
                self.update_position(0.5, 0.5)
        # else:
        #    self.update_position(0.5, 0.5)

    def start_smiling(self, max_duration: float | None = None) -> None:
        if not self.cfg.ignore_all_input:
            self.iris.start_smiling(max_duration)

    def start_crying(self, max_duration: float | None = None) -> None:
        if not self.cfg.ignore_all_input:
            self.iris.start_crying(max_duration)

    def close_eye(self, max_duration: float | None = None) -> None:
        if not self.cfg.ignore_all_input:
            self.iris.close_eye(max_duration=max_duration)

    def close_eye_at_angle(self, max_duration: float | None = None, angle: float | None = None) -> None:
        if not self.cfg.ignore_all_input:
            self.iris.close_eye_at_angle(max_duration=max_duration, angle=angle)

    def refresh_outimg(self) -> None:
        """Get a new eye canvas (e.g., for a new iteration)"""
        self.outimg[:] = self.sclera#.copy()

    def update_visuals(self) -> None:
        """Call functions required for visualization
        in the correct order
        """
        self.refresh_outimg()
        if self.show_iris:
            self.iris.draw(self.outimg)
        if self.show_pupil and not self.pupil_disabled_by_action:
            if self.draw_pupil_behind_reflection:
                self.pupil.draw(self.outimg, self.x, self.y)
        
        if self.show_mirror and not self.mirror_disabled_by_action:
            try:
                self.apply_mirror_mode(allow_mirror=True)
            except ValueError as e:
                logging.error("incompatible mirror img.")
                logging.error(e)

        if not self.draw_pupil_behind_reflection and self.show_pupil and not self.pupil_disabled_by_action:
            self.pupil.draw(self.outimg, self.x, self.y)

        if self.video_recorder.active:
            self.video_recorder.video_frame = self.outimg
            self.video_recorder.update_video_capture()

    def apply_mirror_mode(self, allow_mirror: bool = True) -> None:
        """Construct output image with optional mirroring features"""
        # full bg mirroring is the default
        # the other modes apply masks of varying size
        reflection = self.mirror.reflection 
        if reflection is None:
            return 

        # experimental: apply radial distortion to mirror image
        #reflection = radial_distortion(reflection)

        if self.mirror_on_pupil and not self.mirror_on_iris:
            reflection = apply_blur_circle_mask(
                img=reflection, blur_circle=self.pupil.blur_circle, centerx=self.x, centery=self.y
            )
        elif self.mirror_on_pupil and self.mirror_on_iris:
            reflection = apply_blur_circle_mask(
                img=reflection, blur_circle=self.iris.blur_circle, centerx=self.x, centery=self.y
            )
        elif self.mirror_on_iris and not self.mirror_on_pupil:
            self.pupil.draw(reflection, self.x, self.y)
            reflection = apply_blur_circle_mask(
                img=reflection, blur_circle=self.iris.blur_circle, centerx=self.x, centery=self.y
            )
        if self.show_mirror and allow_mirror:
            self.merge_eye_and_camera(reflection, opacity=self.mirror.opacity)

        # apply frame
        mask = self.frame[..., 3] != 0
        self.outimg[mask] = self.frame[..., :3][mask]

    def merge_eye_and_camera(self, img: cv2.typing.MatLike, opacity: float) -> None:
        """Plot mirror image"""
        try:
            cv2.addWeighted(self.outimg, 1.0, img, opacity, 0, dst=self.outimg)
        except cv2.error as e:
            logging.error(e)
            logging.error(f"Mirror and eye image not compatible. {self.outimg.shape} vs {img.shape} ")

    def build_frame(self, color: tuple[int, int, int, int] = (152, 122, 105, 255), thickness: int = 10) -> None:
        """Builds a frame around the eyes instead of
        loading it from a file
        """
        # w, h = self.frame.shape[:2]
        w, h = self.width, self.height
        canvas = np.zeros((h, w, 4), dtype="uint16")  # +200
        halfthickness = thickness // 2
        cv2.rectangle(
            canvas,
            pt1=(halfthickness, halfthickness),
            pt2=(w - halfthickness, h - halfthickness),
            color=color,
            thickness=thickness,
        )
        canvas = cv2.GaussianBlur(canvas, (21, 21), 10)
        self.frame = canvas

    def build_no_frame(self, width: int = 360, height: int = 360) -> None:
        self.frame = np.zeros((height, width, 4), dtype="uint16")  # +200
