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

from mirror_eyes.custom_types import Extents, ScreenPoint
from mirror_eyes.tools.bounding_box import BoundingBox2D


def roi_to_img_scale(roi: BoundingBox2D, target_width: int, target_height: int):
    """Calculate a scaling factor for a given region of interest
    based on its width and height relative to a target_width and height.
    If the roi exceeds the target on at least one dimension,
    a ratio is returned that specifies how much an image would need to
    be downscaled for the roi to fit into a frame with the given
    target width and target height."""
    if roi is None:
        return 1, 0
    width_missing = max(roi.width - target_width, 0)
    height_missing = max(roi.height - target_height, 0)
    if width_missing >= height_missing:
        missing_pixels = width_missing
        scale = min(target_width / roi.width, 1.0)
    else:
        missing_pixels = height_missing
        scale = min(target_height / roi.height, 1.0)
    return scale, missing_pixels


def downscale_image_around_point(
    img: cv2.typing.MatLike, scale: float, point: ScreenPoint) -> tuple[cv2.typing.MatLike, float, float]:
    """
    Shrinks a given image according to a given scaling factor
    and places it on a canvas of the original image size such that
    the given point on the original image corresponds to the same point on the downscaled image.
    Empty canvas space is filled with repeated boundary pixel copies.
    """
    if not (0 < scale < 1):
        return img, 0.0, 0.0 

    orig_img_height, orig_img_width = img.shape[:2]
    new_img_width = int(orig_img_width * scale)
    new_img_height = int(orig_img_height * scale)

    resized_img = cv2.resize(img, dsize=(new_img_width, new_img_height))

    # calculate where the target point ends up on the resized image 
    point_on_rescaled_image = ScreenPoint(x=int(point.x * scale), y=int(point.y * scale))

    # Calculate the offset needed to keep the point at the original coordinate 
    # Offset = original position - rescaled position 
    x_offset = point.x - point_on_rescaled_image.x
    y_offset = point.y - point_on_rescaled_image.y 

    # by placing the rescaled-image with a scale- and position-determined
    # offset, the center of the roi remains at the same location.
    # use BoundingBox2D to handle boundary corrections and calculate margins 
    offset_bbox = BoundingBox2D(
        top_left=ScreenPoint(x=x_offset, y=y_offset),
        extents=Extents(width=new_img_width, height=new_img_height),
        max_extents=Extents(width=orig_img_width, height=orig_img_height),
    )

    try:
        canvas = cv2.copyMakeBorder(
            resized_img,
            top=offset_bbox.top_margin,
            bottom=offset_bbox.bottom_margin,
            left=offset_bbox.left_margin,
            right=offset_bbox.right_margin,
            borderType=cv2.BORDER_REPLICATE,
        )
    except ValueError as e:
        logging.error("Downscaling failed")
        logging.error(e)
        return img, 0.0, 0.0

    # instead of a edge pixel canvas, consider placing it on the original image
    # img[y_offset:y_offset+new_img_height, x_offset:x_offset+new_img_width] = resized_img
    # return img, x_offset, y_offset
    return canvas, offset_bbox.top_left.x, offset_bbox.top_left.y


def rescale_image_around_point(
    img: cv2.typing.MatLike, scale: float, point: ScreenPoint) -> tuple[cv2.typing.MatLike, float, float]:
    """Downscaling wrapper to simplify integration with upscaling if needed in the future."""
    if scale > 1:
        logging.warning("only downscaling of input image supported")
        return img, 0.0, 0.0
    return downscale_image_around_point(img=img, scale=scale, point=point)


def add_blended_margin(image: cv2.typing.MatLike, margin: int) -> cv2.typing.MatLike:
    """
    Add a matching background margin around an RGB image using edge pixel replication

    Args:
        image (cv2.typing.MatLike): Input RGB image (H X W x 3)
        margin (int): Margin width to be added on each sider of the input image

    Returns:
        cv2.typing.MatLike: Output image with matching margin (H+2*margin x W+2*margin x 3)
    """
    if margin <= 0:
        return image.copy()

    # check if image is RGB
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be RGB with shape (H, W, 3).")

    # use cv2 border pixel replication for margin creation
    return cv2.copyMakeBorder(
        image, top=margin, bottom=margin, left=margin, right=margin, borderType=cv2.BORDER_REPLICATE
    )


def add_smooth_blended_margin(image: cv2.typing.MatLike, margin: int, blur_ksize: int = 15) -> cv2.typing.MatLike:
    """
    Add a matching background margin around an RGB image using edge pixel replication

    Args:
        image (np.ndarray): Input RGB image (H X W x 3)
        margin (int): Margin width to be added on each sider of the input image
        blur_ksize (int): Kernel size for gaussian blur (must be an odd number)

    Returns:
        np.ndarray: Output image with matching margin (H+2*margin x W+2*margin x 3)
    """
    if margin <= 0:
        return image.copy()

    # check if image is RGB
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be RGB with shape (H, W, 3).")

    # use cv2 border pixel replication for margin creation
    expanded_image = cv2.copyMakeBorder(
        image, top=margin, bottom=margin, left=margin, right=margin, borderType=cv2.BORDER_REPLICATE
    )
    h, w = image.shape[:2]
    blur_size = max(3, (margin // 4) * 2 + 1)  # odd number for gaussian kernel
    blurred_image = cv2.GaussianBlur(expanded_image, (blur_size, blur_size), 0)

    blended_image = blurred_image.copy()
    blended_image[margin : margin + h, margin : margin + w] = image

    return blended_image
