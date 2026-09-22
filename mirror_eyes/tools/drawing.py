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
from collections import deque
from time import time

import cv2
import numpy as np

from mirror_eyes.custom_types import Extents, Limits, ScreenPoint

# from mirror_eyes.tools.color import Color

def radial_distortion(img: np.ndarray, k_1: float = 0.2, k_2: float = 0.05) -> np.ndarray:
    """
    Apply Fisheye-like radial distortion.
    Optimized to use cv2.remap

    Adapted from method described here: https://stackoverflow.com/a/68320902
    """
    h, w = img.shape[:2]
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Normalize coordinates to [-1, 1]
    x_c, y_c = w / 2, h / 2
    x_norm = (x - x_c) / x_c
    y_norm = (y - y_c) / y_c
    
    radius_sq = x_norm**2 + y_norm**2
    m_r = 1 + k_1 * np.sqrt(radius_sq) + k_2 * radius_sq
    
    # Apply distortion and denormalize
    map_x = (x_norm * m_r * x_c + x_c).astype(np.float32)
    map_y = (y_norm * m_r * y_c + y_c).astype(np.float32)
    
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)


def line_through_center_at_angle(centerx, centery, angle, length):
    """adapted from https://stackoverflow.com/a/14842362"""
    cosang, sinang = np.cos(angle), np.sin(angle)
    radius = length / 2
    # start from a horizontal line
    x1, y1 = centerx - radius, centery
    x2, y2 = centerx + radius, centery
    tx1, ty1 = x1 - centerx, y1 - centery
    tx2, ty2 = x2 - centerx, y2 - centery
    p1x = (tx1 * cosang + ty1 * sinang) + centerx
    p1y = (-tx1 * sinang + ty1 * cosang) + centery
    p2x = (tx2 * cosang + ty2 * sinang) + centerx
    p2y = (-tx2 * sinang + ty2 * cosang) + centery
    return [int(p1x), int(p1y)], [int(p2x), int(p2y)]


def line_orthogonal_to_center(centerx, centery, angle, distance, length):
    cosang, sinang = np.cos(angle), np.sin(angle)
    # radius = length / 2
    radius = distance
    # start from a horizontal line
    x1, y1 = centerx - radius, centery
    x2, y2 = centerx - radius + length, centery
    tx1, ty1 = x1 - centerx, y1 - centery
    tx2, ty2 = x2 - centerx, y2 - centery
    p1x = (tx1 * cosang + ty1 * sinang) + centerx
    p1y = (-tx1 * sinang + ty1 * cosang) + centery
    p2x = (tx2 * cosang + ty2 * sinang) + centerx
    p2y = (-tx2 * sinang + ty2 * cosang) + centery
    return [int(p1x), int(p1y)], [int(p2x), int(p2y)]


def draw_line_with_corners(img, pt1, pt2, color, thickness):
    """adapted from https://stackoverflow.com/a/73054292"""
    x1, y1, x2, y2 = *pt1, *pt2
    theta = np.pi - np.arctan2(y1 - y2, x1 - x2)
    dx = int(np.sin(theta) * thickness / 2)
    dy = int(np.cos(theta) * thickness / 2)
    points = [[x1 + dx, y1 + dy], [x1 - dx, y1 - dy], [x2 - dx, y2 - dy], [x2 + dx, y2 + dy]]
    cv2.fillPoly(img, [np.array(points)], color)


def apply_round_mask_to_image_leg(img: np.ndarray, centerx: int, centery: int, radius: int = 72) -> np.ndarray:
    """Creates a circular mask (transparent circle on black background)
    with a given radius at a given position
    Computationally this is cheaper than the blur alternative
    defined below. Consider using this on slower hardware if required.

    Args:
    ----
        img (cv2 image) - image to which mask should be applied
        centerx, centery (int) - center of mask circle in img coordinates
        radius (int) - radius of mask circle in pixel

    Returns:
    -------
        input image with circular mask

    """
    cmask = np.zeros_like(img)
    cmask = cv2.circle(cmask, (centerx, centery), radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    # put mask into alpha channel of input
    result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = cmask[:, :, 0]
    # apply mask to image
    result = cv2.bitwise_and(img, cmask)
    return result


def apply_round_mask_to_image(img: np.ndarray, centerx: int, centery: int, radius: int = 72) -> np.ndarray:
    # Use a single channel mask for efficiency
    mask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    cv2.circle(mask, (centerx, centery), radius, 255, -1, lineType=cv2.LINE_AA)
    
    # Convert mask to 3 channels and apply via multiplication
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(img, mask_3ch)
    

def make_blur_circle(radius: int, blurwidth: int = 10) -> np.ndarray:
    """Partial substitution for apply_round_blur_mask_to_image()
    that is just called once during initialization and thus
    avoids expensive gaussian blur applications in the main loop.
    Should be used in combination with apply_blur_circle_mask()

    Args:
    ----
        radius (int) - radius of transparent circle given in pixel
        blurwidth (int) - additional radius to be used for a blurred
            area around the circle to create a fading effect

    Returns:
    -------
        mask consisting of a black area with a blurred transparent
        circle in the center

    """
    radius = radius + 3 * blurwidth  # *2# + blurwidth
    mask = np.zeros((2 * radius, 2 * radius, 3), dtype="uint8")
    cv2.circle(mask, (radius, radius), radius - 2 * blurwidth, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (41, 41), radius)
    logging.info(f"Created blur mask with shape {mask.shape}")
    # dimensions check not required in this case
    # if mask.ndim==3 and mask.shape[-1] == 3:
    # alpha = mask/255.0
    return mask


def apply_blur_circle_mask(img: np.ndarray, blur_circle: np.ndarray, centerx: int, centery: int):
    """Applies a previously generated blur_circle/mask to a given
    image at a given position.
    To stay within the boundaries defined by the input image
    resolution, cropping of the blur circle is applied when necessary.

    Args:
    ----
        img (cv2 image) - image to which mask should be applied
        blur_circle (numpy array) - blurred circle mask generated by
            make_blur_circle()
        centerx, centery (int) - center of mask circle in img coordinates

    Returns:
    -------
        input image with applied blurry transparent circle
        at position centerx, centery and black outside that
        circle.

    """
    # create mask canvas
    mask = np.zeros_like(img)  # , dtype = "uint8"
    # some cropping needs to be applied in the corners because the blur
    # area is larger than the iris
    h, w = blur_circle.shape[:2]
    my, mx = mask.shape[:2]
    halfh = h // 2
    halfw = w // 2
    y_crop_upper = centery + halfh - my
    y_crop_lower = centery - halfh
    x_crop_upper = centerx + halfw - mx
    x_crop_lower = centerx - halfw
    mask_y_lower = max(0, y_crop_lower)
    mask_y_upper = min(my, centery + halfh)
    mask_x_lower = max(0, x_crop_lower)
    mask_x_upper = min(mx, centerx + halfw)
    circle_y_lower = min(y_crop_lower, 0) * -1
    circle_y_upper = min(h - y_crop_upper, h)
    circle_x_lower = min(x_crop_lower, 0) * -1
    circle_x_upper = min(w - x_crop_upper, w)
    # put blurred circle on the mask
    mask[mask_y_lower:mask_y_upper, mask_x_lower:mask_x_upper] = blur_circle[
        circle_y_lower:circle_y_upper, circle_x_lower:circle_x_upper, :3
    ]
    alpha = mask / 255.0
    return cv2.convertScaleAbs(img * alpha)


def apply_round_blur_mask_to_image(
    img: np.ndarray, centerx: int, centery: int, radius: int = 72, blurwidth: int = 10
) -> np.ndarray:
    """Legacy function

    Expensive on-demand one step generation of blurry transparent circle
    with a given radius and blurwidth that is placed at a
    given location (centerx, centery) on an image. The rest of the
    image is turned dark.
    This can substitute apply_blur_circle_mask() and make_blur_circle()
    But is computationally more expensive and should hence be avoided
    unless, e.g., mask size should change dynamically.
    In most cases the combination of apply_blur_circle_mask() and
    make_blur_circle() as used in EyeModel is preferable.

    Args: see combined args of apply_blur_circle_mask() and make_blur_circle()

    Returns
    -------
        input image with applied blurry transparent circle
        of a given radius at position centerx, centery and
        black outside that circle.

    """
    radius += blurwidth

    # option: apply a small offset from center to simulate light coming from
    # a specific direction (to increase appearance as reflection)
    # might not be necessary because objects become brighter, hence more
    # visible, in image parts with stronger illumination. This creates a
    # natural offset.
    # centerx += 5
    # centery -= 5
    # build mask
    mask = np.zeros_like(img)
    cv2.circle(mask, (centerx, centery), radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    # apply blur (WARNING: costly operation, especially
    # with larger numbers in second argument)
    mask = cv2.GaussianBlur(mask, (41, 41), radius)
    # dimensions check not required in this case
    # if mask.ndim==3 and mask.shape[-1] == 3:
    alpha = mask / 255.0
    # else:
    #    alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    # result = cv2.convertScaleAbs(img*(1-alpha))
    # blending
    result = cv2.convertScaleAbs(img * alpha)
    return result


def add_img_with_transparency(img: np.ndarray, bgimg: np.ndarray):
    """Put an image with a transparent
    component onto another image
    """
    alpha_s = img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(3):
        bgimg[:, :, c] = alpha_s * img[:, :, c] + alpha_l * bgimg[:, :, c]
    return bgimg


class RotatingImage:
    def __init__(self, image=None) -> None:
        if image is not None:
            self.update_image(image)
        else:
            logging.warning("Image required for rotation to be possible")
        self.last_rotation_time = time()
        self.step_length = 0.0
        self._orientation_deg = 0.0
        self.rotation_velocity = 0

    @property
    def rotating(self):
        return self.rotation_velocity != 0

    @property
    def img(self):
        now = time()
        if self.rotating and (now - self.last_rotation_time >= self.step_length):
            self.update_orientation()
            self.rotate()
            self.last_rotation_time = now
        return self._image

    @property
    def orientation_deg(self) -> int:
        return int(self._orientation_deg)

    def update_orientation(self) -> None:
        self._orientation_deg = (self._orientation_deg + self.rotation_velocity) % 360

    def update_image(self, image):
        self._image = image
        self.original_image = image[:]
        self.set_extents()

    def set_extents(self, image=None):
        if image is not None:
            height, width, _ = image.shape
        else:
            height, width, _ = self._image.shape
        self.extents = Extents(width=width, height=height)
        self.center = ScreenPoint(x=width // 2, y=height // 2)

    def rotate(self):
        m = cv2.getRotationMatrix2D((self.center.x, self.center.y), self.orientation_deg, 1.0)
        self._image = cv2.warpAffine(self.original_image, m, (self.extents.width, self.extents.height))

    def start_rotation_at_degrees_per_second(self, degrees_per_second: int, rotation_velocity: float = 1):
        self.rotation_velocity = rotation_velocity
        if degrees_per_second < 0:
            self.rotation_velocity *= -1
        self.step_length = rotation_velocity / abs(degrees_per_second)

    def stop_rotation(self):
        self.rotation_velocity = 0


class ScalableRotatingImage(RotatingImage):
    def __init__(self, image, minimum_size: int = 70) -> None:
        super().__init__(image=image)
        self.size_limits = Limits(min=minimum_size, max=self.extents.width)
        self.size = self.size_limits.max
        self.resized_image = self.original_image[:]
        self.size_updated = True
        self.resizing = False
        self.last_resize_time = time()

    @property
    def img(self):
        now = time()
        if self.resizing and self.size_target_reached:
            self.resizing = False
            self.reset_size()
            self.update_orientation()
            self.rotate()
            self.stop_rotation()
            self.size_updated = True
        if self.resizing and (now - self.last_resize_time) >= self.resize_step_duration:
            self.update_size()
            self.size_updated = True
            self.last_resize_time = now
        time_for_a_new_rotation = self.rotating and (now - self.last_rotation_time >= self.step_length)
        if (self.size_updated and self.rotating) or time_for_a_new_rotation:
            self.update_orientation()
            self.rotate()
            if time_for_a_new_rotation:
                self.last_rotation_time = now
        self.size_updated = False
        return self._image

    @property
    def size_target_reached(self):
        if not self.resizing:
            return True
        if self.resize_step_size < 0:
            return self.size_limits.min >= self.size
        else:
            return self.size_limits.max <= self.size

    def rotate(self):
        m = cv2.getRotationMatrix2D((self.center.x, self.center.y), self.orientation_deg, 1.0)
        self._image = cv2.warpAffine(self.resized_image, m, (self.extents.width, self.extents.height))

    def update_size(self):
        new_size = min(max(self.size + self.resize_step_size, self.size_limits.min), self.size_limits.max)
        self.resized_image = cv2.resize(self.original_image, dsize=(new_size, new_size))
        if not self.rotating:
            self._image = self.resized_image
        self.size = new_size
        self.set_extents(image=self.resized_image)

    def start_resizing(self, shrink: int = True, duration_seconds: float = 0.25, step_number: int = 5):
        self.resizing = True
        self.resize_step_duration = duration_seconds / step_number
        if shrink:
            self.resize_step_size = -1 * (self.size - self.size_limits.min) // step_number
        else:
            self.resize_step_size = (self.size_limits.max - self.size) // step_number

    def start_resizing_and_rotating(
        self, shrink: int = True, duration_seconds: float = 0.5, step_number: int = 15, degrees_per_second: int = -180
    ):
        self.start_resizing(shrink=shrink, duration_seconds=duration_seconds, step_number=step_number)
        self.start_rotation_at_degrees_per_second(degrees_per_second=degrees_per_second, rotation_velocity=5)

    def reset_size(self):
        self.resizing = False
        self._image = self.original_image[:]
        self.resized_image = self._image
        self.size = self.size_limits.max
        self.set_extents(image=self._image)


class LoadingCircle:
    def __init__(
        self,
        radius: int,
        lines: int = 32,
        brightness_limits: Limits = Limits(min=0, max=24),
        direction: int = 1,
        cycle_duration: float = 3.0,
    ) -> None:
        self.radius = radius
        self.extents = Extents(width=radius * 2, height=radius * 2)
        self.lines = lines
        self.brightness_limits = brightness_limits
        self.direction = direction
        self.set_cycle_duration(cycle_duration)
        self.last_step = time()
        self.state_idx = 0
        self.build_base_state()
        self.build_state_images()

    @property
    def img(self):
        now = time()
        if now - self.last_step >= self.step_length:
            self.state_idx = (self.state_idx + 1) % self.lines
            self.last_step = now
        return self.state_images[self.state_idx]

    def synchronize_state(self, state_idx, last_step=None):
        self.state_idx = state_idx % self.lines
        if last_step is not None:
            self.last_step = last_step

    def set_cycle_duration(self, cycle_duration: float) -> None:
        self.cycle_duration = cycle_duration
        self.step_length = cycle_duration / self.lines

    def build_base_state(self):
        float_states_linear = np.arange(0, 1, 1 / self.lines)
        float_states_exp = np.power(float_states_linear, 2) * self.brightness_limits.max
        float_states = float_states_exp.astype(np.int32)
        self.base_state = deque(float_states.tolist())

    def draw_beams_state(self):
        canvas = np.zeros((self.extents.height, self.extents.width, 4), dtype="uint8")

        for i, angle in enumerate(np.arange(0, np.pi * 2, (np.pi * 2) / self.lines)):
            p1, p2 = line_orthogonal_to_center(
                centerx=self.radius, centery=self.radius, angle=angle, length=self.radius, distance=self.radius
            )
            beamcolor = 3 * (self.base_state[i],)
            cv2.line(canvas, p1, p2, beamcolor, thickness=16)
        return canvas

    def build_state_images(self):
        self.state_images = {}  # a state is a list of ints that
        # indicate how much brightness should be added to each
        # line
        for i in range(self.lines):
            self.base_state.rotate(self.direction)
            self.state_images[i] = self.draw_beams_state()


def filter_gray(img: cv2.typing.MatLike):
    """Convert to grayscale"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.merge([gray, gray, gray])


def filter_edge(img: cv2.typing.MatLike):
    """Apply canny edge detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 50, 200)
    return cv2.merge([edge, edge, edge])


def filter_thresh(img: cv2.typing.MatLike, color=True):
    """Apply thresholding"""
    if not color:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grey, 127, 255, cv2.THRESH_TRUNC)
        return cv2.merge([thresh, thresh, thresh])
    else:
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
        return thresh


def filter_thresh_color(img: cv2.typing.MatLike):
    """Color thresholding"""
    return filter_thresh(img, color=True)


def filter_thresh_sw(img: cv2.typing.MatLike):
    """Greyscale thresholding"""
    return filter_thresh(img, color=False)


def filter_thresh_bin(img: cv2.typing.MatLike):
    """Create binary image"""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)
    return cv2.merge([thresh, thresh, thresh])


def filter_thresh_bin_adaptive(img: cv2.typing.MatLike):
    """Binary image with adaptive threshold"""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10
    )  # ADAPTIVE_THRESH_GAUSSIAN_C
    return cv2.merge([thresh, thresh, thresh])


def filter_blur(img: cv2.typing.MatLike):
    return cv2.blur(img, (21, 21))


def filter_blur_thresh_color(img: cv2.typing.MatLike):
    return cv2.blur(filter_thresh_color(img), (21, 21))


def filter_default(img: cv2.typing.MatLike):
    return cv2.blur(filter_thresh_color(img), (11, 11))


def filter_nothing(img: cv2.typing.MatLike):
    return img


def filter_smooth_edge(img: cv2.typing.MatLike):
    return cv2.GaussianBlur(filter_edge(img), (15, 15), 0)


fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)


def filter_motion_detection(img: cv2.typing.MatLike):
    fg = fgbg.apply(img)
    return cv2.bitwise_and(cv2.merge([fg, fg, fg]), img)


image_filters = [
    filter_nothing,
    filter_default,
    filter_blur,
    filter_gray,
    filter_thresh_color,
    filter_blur_thresh_color,
    filter_thresh_sw,
    filter_thresh_bin,
    filter_thresh_bin_adaptive,
    filter_edge,
    filter_motion_detection,
]
