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
from mirror_eyes.tools.bounding_box import BoundingBox2D, SmoothBoundingBox2D
import pytest

from mirror_eyes.custom_types import ScreenPoint, Extents

positive_point = ScreenPoint(x=10, y=10)
negative_x_point = ScreenPoint(x=-10, y=10)
negative_y_point = ScreenPoint(x=10, y=-10)
negative_point = ScreenPoint(x=-20, y=-20)

positive_extents = Extents(width=30, height=30)
negative_y_extents = Extents(width=30, height=-30)
max_extents = Extents(width=100, height=100)
too_large_extents = Extents(width=101, height=101)


@pytest.mark.parametrize(
    ("top_left", "extents", "max_extents"),
    [
        pytest.param(
            positive_point,
            positive_extents,
            max_extents,
            id="Bounding Box within maximum extents",
        ),
        pytest.param(
            negative_x_point,
            positive_extents,
            max_extents,
            id="Bounding Box starts left of maximum extents (negative x)",
        ),
        pytest.param(
            negative_y_point,
            positive_extents,
            max_extents,
            id="Bounding Box starts above maximum extents (negative y)",
        ),
        pytest.param(
            negative_point,
            positive_extents,
            max_extents,
            id="Bounding Box starts left and above of maximum extents (negative x,y)",
        ),
        pytest.param(
            positive_point,
            too_large_extents,
            max_extents,
            id="Bounding Box larger than maximum extents",
        ),
        pytest.param(
            positive_point,
            max_extents,
            max_extents,
            id="Bounding Box as large as maximum extents but above limits",
        ),
        pytest.param(
            positive_point, too_large_extents, None, id="No limits for bounding box"
        ),
    ],
)
def test_bounding_box_boundary_correction(top_left, extents, max_extents):
    bounding_box = BoundingBox2D(
        top_left=top_left, extents=extents, max_extents=max_extents
    )
    # print(f"tl before: {bounding_box.top_left} ")
    # print(f"br before: {bounding_box.bottom_right}\n")
    bounding_box.update(top_left=top_left, extents=extents, boundary_correction=True)
    # print(f"tl after: {bounding_box.top_left} ")
    # print(f"br after: {bounding_box.bottom_right}\n")
    assert bounding_box.extents_within_valid_range() == True
    assert bounding_box.bounding_box_within_valid_range() == True

    # repeat for the smooth bounding box variant
    bounding_box_smooth = SmoothBoundingBox2D(
        top_left=top_left, extents=extents, max_extents=max_extents
    )
    bounding_box_smooth.update(
        top_left=top_left, extents=extents, boundary_correction=True
    )
    assert bounding_box_smooth.extents_within_valid_range() == True
    assert bounding_box_smooth.bounding_box_within_valid_range() == True


@pytest.mark.parametrize(
    ("top_left", "extents", "max_extents"),
    [
        pytest.param(
            positive_point,
            positive_extents,
            max_extents,
            id="Bounding Box within maximum extents",
        ),
        pytest.param(
            negative_x_point,
            positive_extents,
            max_extents,
            id="Bounding Box starts left of maximum extents (negative x)",
        ),
        pytest.param(
            negative_y_point,
            positive_extents,
            max_extents,
            id="Bounding Box starts above maximum extents (negative y)",
        ),
        pytest.param(
            negative_point,
            positive_extents,
            max_extents,
            id="Bounding Box starts left and above of maximum extents (negative x,y)",
        ),
        pytest.param(
            positive_point,
            too_large_extents,
            max_extents,
            id="Bounding Box larger than maximum extents",
        ),
        pytest.param(
            positive_point,
            max_extents,
            max_extents,
            id="Bounding Box as large as maximum extents but above limits",
        ),
        pytest.param(
            positive_point, too_large_extents, None, id="No limits for bounding box"
        ),
    ],
)
def test_bounding_box_and_smooth_bounding_box_consistency(
    top_left, extents, max_extents
):
    bounding_box = BoundingBox2D(
        top_left=top_left, extents=extents, max_extents=max_extents
    )
    bounding_box_smooth = SmoothBoundingBox2D(
        top_left=top_left, extents=extents, max_extents=max_extents
    )

    assert bounding_box_smooth.top_left == bounding_box.top_left
    assert bounding_box_smooth.extents == bounding_box.extents
    assert bounding_box_smooth.top_right == bounding_box.top_right
    assert bounding_box_smooth.bottom_left == bounding_box.bottom_left
    assert bounding_box_smooth.bottom_right == bounding_box.bottom_right
    assert bounding_box_smooth.center == bounding_box.center
    assert bounding_box_smooth.height == bounding_box.height
    assert bounding_box_smooth.width == bounding_box.width
    assert (
        bounding_box_smooth.extents_within_valid_range()
        == bounding_box.extents_within_valid_range()
    )
    assert (
        bounding_box_smooth.bounding_box_within_valid_range()
        == bounding_box.bounding_box_within_valid_range()
    )

    # repeat after update
    bounding_box.update(top_left=top_left, extents=extents, boundary_correction=True)
    bounding_box_smooth.update(
        top_left=top_left, extents=extents, boundary_correction=True
    )
    assert bounding_box_smooth.top_left == bounding_box.top_left
    assert bounding_box_smooth.extents == bounding_box.extents
    assert bounding_box_smooth.top_right == bounding_box.top_right
    assert bounding_box_smooth.bottom_left == bounding_box.bottom_left
    assert bounding_box_smooth.bottom_right == bounding_box.bottom_right
    assert bounding_box_smooth.center == bounding_box.center
    assert bounding_box_smooth.height == bounding_box.height
    assert bounding_box_smooth.width == bounding_box.width
    assert (
        bounding_box_smooth.extents_within_valid_range()
        == bounding_box.extents_within_valid_range()
    )
    assert (
        bounding_box_smooth.bounding_box_within_valid_range()
        == bounding_box.bounding_box_within_valid_range()
    )
