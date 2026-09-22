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
from time import time

# external
import cv2
import numpy as np
import screeninfo

# own
from mirror_eyes.config.config import Config, MirrorEyesConfig
from mirror_eyes.custom_types import Limits, ScreenPoint
from mirror_eyes.eye_model import EyeModel
from mirror_eyes.tools.bounding_box import BoundingBox2D, RoiScanIndexer
from mirror_eyes.tools.converting import normalize, remap_coordinates, restrict_to_range
from mirror_eyes.tools.image_scaling import rescale_image_around_point, roi_to_img_scale
from mirror_eyes.tools.video_recording import SingleScreenMirrorEyesRecorder, VideoRecorder

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(filename="debug.log", mode="w"),  # mode w to override previous runs
        logging.StreamHandler(),
    ],
)


def get_screen_position(monitor_name: str, offset: ScreenPoint) -> ScreenPoint:
    """
    Looks for a monitor based on its name and sets a screen position relative to it.

    Args:
        monitor_name: monitor identifier
        offset: offset that should be added to the screen postion

    Returns:
        ScreenPoint: screen position + offset

    """
    if not monitor_name:
        return offset

    # Find the given monitor and work relative to it.
    monitors = screeninfo.get_monitors()
    for monitor in monitors:
        if monitor.name == monitor_name:
            return ScreenPoint(x=monitor.x + offset.x, y=monitor.y + offset.y)
    else:
        raise ValueError(
            f"There is no monitor with name '{monitor_name}'. "
            f"Available monitors are {[monitor.name for monitor in monitors]}."
        )


class MirrorEyes:
    """Camera-agnostic implementation of virtual mirroring on multiple eye models.
    Two eyes can be displayed simultaneously with distinct spatial offsets
    to simulate binocular disparity.

    Requires integration with
    1. a module that supplies an image to be shown
    2. an attention module to supply a point of interest (on the above image)
    3. (optional) an expression module that manages appearance aspects of the eyes and mirrors (e.g., opacity)
    """

    def __init__(
        self,
        config: MirrorEyesConfig,
        body = None,
        video_recorder: VideoRecorder | None = None,
    ) -> None:
        """
        Args:
            config: mirror eyes configuration
            body: optional body to be displayed together with the eyes
            video_recorder: optional video recording object to create videos of the eyes when executed
            show_fps: optional display of frames per second

        """
        self.setup_complete = False
        self.eyes = [EyeModel(eye_cfg=eye_config) for eye_config in config.eyes]
        self.body = body
        self.cfg = config
        self.display_cfg = config.display
        self.show_fps = self.display_cfg.show_fps
        self.last_fps_check_time = time()
        self.init_canvas()

        if video_recorder is not None:
            self.video_recorder = video_recorder
        else:
            self.video_recorder = SingleScreenMirrorEyesRecorder(canvas=self.canvas, active=False, fps=30)
        self.init_time = time()

        self._img_scale = 0
        self._img_scale_based_offset = ScreenPoint(x=0, y=0)
        self.scale_smoothing_factor = 1.0 / self.eyes[0].mirror.cfg.scale_hysteresis \
            if self.eyes[0].mirror.cfg.scale_hysteresis > 0 else 1.0 

        self.roi_scan_simulator = RoiScanIndexer()
        logging.info("Initialized mirror eyes")

    @property
    def img_scale(self) -> float:
        """Get image scale"""
        return self._img_scale

    @img_scale.setter
    def img_scale(self, value: float) -> None:
        """Set image scale with smoothing"""
        self._img_scale = (value * self.scale_smoothing_factor) + \
            (self._img_scale * (1.0 - self.scale_smoothing_factor))


    @property
    def img_scale_based_offset(self) -> ScreenPoint:
        return self._img_scale_based_offset

    @img_scale_based_offset.setter
    def img_scale_based_offset(self, new_offset: ScreenPoint) -> None:
        self._img_scale_based_offset = ScreenPoint.from_array(
            (new_offset * self.scale_smoothing_factor) + \
                (self._img_scale_based_offset * (1 - self.scale_smoothing_factor))
        )


    def init_canvas(self) -> None:
        self.canvas = np.zeros(
            (self.display_cfg.size_pixel.height, self.display_cfg.size_pixel.width, 3), dtype="uint8"
        )

    def setup(self) -> None:
        if not self.setup_complete:
            if self.body is not None:
                self.body.show()
            # initialize windows for eyes
            cv2.namedWindow("eye_window", flags=self.display_cfg.window_flags) # | cv2.WINDOW_GUI_EXPANDED

            screen_position = get_screen_position(self.display_cfg.screen_name, self.display_cfg.screen_position)
            logging.info(f"Using screen position {screen_position}")
            cv2.moveWindow("eye_window", screen_position.x, screen_position.y)

            if len(self.display_cfg.window_properties) % 2:
                raise ValueError("Length of window properties must be even.")

            for i in range(len(self.display_cfg.window_properties) // 2):
                cv2.setWindowProperty(
                    "eye_window",
                    self.display_cfg.window_properties[i * 2],
                    self.display_cfg.window_properties[i * 2 + 1],
                )

            scale = self.display_cfg.window_scale
            # if scale != 1.0:  # apply scale in any case to overwrite opencv 
            # windows registry entry from prior run
            logging.info(f"scaling window by x{scale} according to configuration.")
            cv2.resizeWindow(
                "eye_window", 
                int(self.display_cfg.size_pixel.width * scale), 
                int(self.display_cfg.size_pixel.height * scale))

            self.setup_complete = True

    def start_roi_scanning(
        self, 
        max_duration: float | None = None, 
        focus_point_duration_range: Limits | None = None) -> None:
        self.roi_scan_simulator.start(max_duration=max_duration, focus_point_duration_range=focus_point_duration_range)

    def fps(self) -> float:
        """just call this once per loop."""
        now = time()
        fps = round(1 / (now - self.last_fps_check_time), 0)
        self.last_fps_check_time = now
        return fps

    def canvas_coordinates(self, eye: EyeModel, from_center: bool = True) -> tuple[int, int, int, int]:
        """
        Args:
          eye (EyeModel)
          from_center (bool): True if canvas coordinates are specified from its center
              False if they specify the upper left corner
        """
        halfheight = int(eye.height / 2)
        halfwidth = int(eye.width / 2)
        loc = eye.canvas_location

        if from_center:
            y1 = loc.y - halfheight
            y2 = loc.y + halfheight
            x1 = loc.x - halfwidth
            x2 = loc.x + halfwidth
        else:
            y1 = loc.y
            y2 = loc.y + eye.height
            x1 = loc.x
            x2 = loc.x + eye.width

        # enforce coordinate limits at display boundaries
        if y1 < 0:
            y2 = y2 - y1
            y1 = 0
        if y2 > self.display_cfg.size_pixel.height:
            y1 = self.display_cfg.size_pixel.height - eye.height  # -1
            y2 = self.display_cfg.size_pixel.height  # -1
        if x1 < 0:
            x2 = x2 - x1
            x1 = 0
        if x2 > self.display_cfg.size_pixel.width:
            x1 = self.display_cfg.size_pixel.width - eye.width  # -1
            x2 = self.display_cfg.size_pixel.width  # - 1
        return y1, y2, x1, x2


    def update_canvas(self, from_center: bool = True) -> None:
        self.canvas.fill(0)
        for eye in self.eyes:
            y1, y2, x1, x2 = self.canvas_coordinates(eye, from_center=from_center)
            self.canvas[y1:y2, x1:x2] = eye.outimg

    def show_visuals(self) -> None:
        if self.show_fps:
            fps = self.fps()
            cv2.putText(
                self.canvas, f"FPS: {fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1
            )
        cv2.imshow("eye_window", self.canvas)

    def roi_based_image_rescaling(
        self, poi: tuple[int, int, float], img_frame: cv2.typing.MatLike, roi: BoundingBox2D
    ) -> tuple[cv2.typing.MatLike, float, float]:
        if roi is None:
            return img_frame, 0, 0
        # x_camera, y_camera, _ = poi

        self.img_scale, _ = roi_to_img_scale(
            roi=roi, target_width=self.eyes[0].cfg.size.width, target_height=self.eyes[1].cfg.size.height
        )
        return rescale_image_around_point(img_frame, self.img_scale, roi.center)

    def update_eyes(
        self,
        img_frame: cv2.typing.MatLike,
        poi: tuple[int, int, float],
        opacity: float = 0.5,
        eye_positions: tuple[float, float, float, float] | None = None,
        roi: BoundingBox2D | None = None,
    ):
        """

        Args:
            eye_positions: [x, y, z, focus_distance] specified in meters
        """
        x_camera, y_camera, distance = poi

        if self.roi_scan_simulator.active and roi is not None:
            # if self.scanning_roi and roi is not None:
            # new_poi = roi.inner_box.point_from_id(self.focus_point_indices[0])
            new_poi = roi.inner_box().point_from_id(self.roi_scan_simulator.focus_point_indices[0])
            x_camera = new_poi.x
            y_camera = new_poi.y

        if self.eyes[0].mirror.cfg.scale_to_roi:
            img_frame, x_offset, y_offset = self.roi_based_image_rescaling(poi, img_frame, roi=roi)
            self.img_scale_based_offset = ScreenPoint(x=x_offset, y=y_offset)

            for eye in self.eyes:
                eye.update_img_scale(self.img_scale, self.img_scale_based_offset)

        for i, eye in enumerate(self.eyes):
            if not eye.mirror.cfg.external_control or eye_positions is None:
                x_on_eye_canvas = None
                y_on_eye_canvas = None
                roi = None
            else:
                # positions on face canvas
                x_on_face_canvas = remap_coordinates(
                    val_source=eye_positions[i][0],
                    min_source=self.display_cfg.size_meters.width / 2,  # meters
                    max_source=-0.5 * self.display_cfg.size_meters.width,  # meters
                    # max_source=self.display_cfg.SIZE_METERS.width,  # meters
                    min_target=0,
                    max_target=self.display_cfg.size_pixel.width,
                )
                y_on_face_canvas = remap_coordinates(
                    val_source=eye_positions[i][1],
                    min_source=-0.5 * self.display_cfg.size_meters.height,  # meters
                    # min_source=self.display_cfg.SIZE_METERS.height,  # meters
                    max_source=self.display_cfg.size_meters.height / 2,  # meters
                    min_target=0,
                    max_target=self.display_cfg.size_pixel.height,
                )

                # positions on eye-canvas
                # convert from meter to pixel
                x_on_eye_canvas = remap_coordinates(
                    val_source=x_on_face_canvas,  # face canvas pixel
                    min_source=eye.cfg.screen_x_range_pixel.min,  # canvas pixel
                    max_source=eye.cfg.screen_x_range_pixel.max,  # canvas pixel
                    min_target=0,  # eye pixel
                    max_target=eye.width,  # eye pixel
                )
                y_on_eye_canvas = remap_coordinates(
                    val_source=y_on_face_canvas,  # pixel face canvas
                    min_source=0,  # pixel
                    max_source=self.display_cfg.size_pixel.height,  # pixel face canvas
                    min_target=0,  # pixel
                    max_target=eye.height,  # pixel
                )
                eye.canvas_location = ScreenPoint(x=x_on_face_canvas, y=y_on_face_canvas)

                distance = restrict_to_range(
                    normalize(
                        val_source=eye_positions[i][3],
                        min_source=eye.cfg.distance_range_meters.min,  # meters
                        max_source=eye.cfg.distance_range_meters.max,  # meters
                        min_target=0.0,
                        max_target=1.0,
                    ),
                    0.0,
                    1.0,
                )
            eye.update_model(
                camera_image=img_frame, #.copy(),
                x_camera=x_camera,
                y_camera=y_camera,
                distance=distance,
                opacity=opacity,
                x_on_eye=x_on_eye_canvas,
                y_on_eye=y_on_eye_canvas,
            )
            eye.update_visuals()

    def start_crying(self, max_duration=None) -> None:
        for eye in self.eyes:
            eye.start_crying(max_duration=max_duration)

    def start_smiling(self, max_duration=None) -> None:
        for eye in self.eyes:
            eye.start_smiling(max_duration=max_duration)

    def close_eyes(self, max_duration=None) -> None:
        for eye in self.eyes:
            eye.close_eye(max_duration=max_duration)

    def close_eyes_at_angle(self, max_duration=None, angle=None) -> None:
        for i, eye in enumerate(self.eyes):
            if angle is not None:
                if i == 1:
                    eye.close_eye_at_angle(max_duration=max_duration, angle=angle)
                else:
                    eye.close_eye_at_angle(max_duration=max_duration, angle=-1 * angle)
            else:
                eye.close_eye(max_duration=max_duration)

    def flirt(self, max_duration=None) -> None:
        for i, eye in enumerate(self.eyes):
            if i == 0:
                eye.close_eye(max_duration=max_duration)
            else:
                eye.start_smiling(max_duration=max_duration)

    def update_camera_resolution(self, width: int, height: int) -> None:
        for eye in self.eyes:
            eye.update_camera_resolution(width=width, height=height)

    def update_pupil_size(self, radius: int) -> None:
        for eye in self.eyes:
            if not eye.cfg.ignore_all_input:
                eye.pupil.radius = radius

    def update_pupil_size_by(self, delta=5) -> None:
        for eye in self.eyes:
            if not eye.cfg.ignore_all_input:
                eye.pupil.radius += delta

    def change_eye_color(self, color: tuple[int, int, int, int], duration: float = 0.0) -> None:
        for eye in self.eyes:
            eye.change_eye_color(color=color, duration=duration)

    def rotate_iris(self, degrees_per_second: int, flip_direction_for_second_eye: bool = True) -> None:
        for i, eye in enumerate(self.eyes):
            if i == 1 and flip_direction_for_second_eye:
                degrees_per_second *= -1
            eye.rotate_iris_at_degrees_per_second(degrees_per_second=degrees_per_second)

    def stop_iris_rotation(self) -> None:
        for eye in self.eyes:
            eye.stop_iris_rotation()

    def show_loading_animation(
        self, 
        cycle_duration: float | None = None, 
        direction: int | None = None) -> None:
        for eye in self.eyes:
            eye.show_loading_animation(cycle_duration=cycle_duration, direction=direction)
        eye_0_loading_state_idx, eye_0_loading_last_step = self.eyes[0].loading_animation_state()
        for eye in self.eyes[1:]:
            eye.synchronize_loading_animation(
                state_idx=eye_0_loading_state_idx, last_step=eye_0_loading_last_step)

    def showing_loading_animation(self) -> bool:
        return self.eyes[0].iris.loading_animation_active

    def stop_loading_animation(self) -> None:
        for eye in self.eyes:
            eye.stop_loading_animation()

    def change_reflection_filter(self, reflection_filter) -> None:
        for eye in self.eyes:
            eye.mirror.reflection_filter = reflection_filter

    def process_img_frame(self, img_frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        # carry out size check
        height, width, *_channels = img_frame.shape
        for eye in self.eyes:
            if height < eye.cfg.mirror.size.height or width < eye.cfg.mirror.size.width:
                logging.warning(
                    f"input frame dimensions must be at least {eye.cfg.mirror.size.height}x{eye.cfg.mirror.size.width}"
                )
                return np.zeros((eye.cfg.mirror.size.height, eye.cfg.mirror.size.width, 3), dtype="uint8")
        if self.eyes[0].mirror.cfg.external_control:
            return img_frame  # cv2.flip(img_frame, 1)
        else:
            return img_frame

    def update(
        self,
        img_frame: cv2.typing.MatLike,
        poi: tuple[int, int, float],
        opacity: float = 1.0,
        eye_positions: tuple[float, float, float, float] | None = None,
        roi: BoundingBox2D | None = None,
    ) -> None:
        img_frame = self.process_img_frame(img_frame)
        self.update_eyes(img_frame, poi, opacity, eye_positions=eye_positions, roi=roi)
        self.update_canvas(from_center=self.eyes[0].mirror.cfg.external_control)
        self.show_visuals()
        self.video_recorder.update_video_capture_from_canvas(canvas=self.canvas)

    def shutdown(self) -> None:
        cv2.destroyAllWindows()
        for eye in self.eyes:
            eye.shutdown()
        self.video_recorder.shutdown()
        logging.info("Shutdown complete")


def main_cfg() -> MirrorEyes:
    # example for setup with configuration
    Config().set_config("local_eye_control")  #'external_eye_control')
    cfg = Config().get_config()
    me = MirrorEyes(config=cfg)
    return me


# if __name__ == "__main__":
# me = main_cfg()
