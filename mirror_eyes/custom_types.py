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

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass
class Point:
    x: float
    y: float
    z: float = 0

    @classmethod
    def from_array(cls, array: np.ndarray) -> Point:
        """
        Creates a Point directly from a numpy array or list.
        Expects the array to have at least three elements [x, y, z].
        """
        # The * operator unpacks the array to x, y, z
        return cls(*array[:3])

    def __array__(self, dtype=None):
        """Tell numpy how to treat it as an array"""
        return np.array([self.x, self.y, self.z], dtype=dtype)

    # enable basic arithmetic 
    def __add__(self, other: ScreenPoint | np.ndarray | list | tuple | float | int) -> ScreenPoint:
        if isinstance(other, ScreenPoint):
            return ScreenPoint(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, (np.ndarray, list, tuple)) and len(other) >= 3:
            return ScreenPoint(self.x + other[0], self.y + other[1], self.z + other[2])
        elif isinstance(other, (int, float)):
            return ScreenPoint(self.x + other, self.y + other, self.z + other)
        return NotImplemented

    def __sub__(self, other: ScreenPoint | np.ndarray | list | tuple | float | int) -> ScreenPoint:
        if isinstance(other, ScreenPoint):
            return ScreenPoint(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, (np.ndarray, list, tuple)) and len(other) >= 3:
            return ScreenPoint(self.x - other[0], self.y - other[1], self.z - other[2])
        elif isinstance(other, (int, float)):
            return ScreenPoint(self.x - other, self.y - other, self.z - other)
        return NotImplemented

    def __mul__(self, scalar: int | float) -> ScreenPoint:
        """Handles multiplication by a scalar (e.g., scaling the point)."""
        if isinstance(scalar, (int, float)):
            return ScreenPoint(self.x * scalar, self.y * scalar, self.z * scalar)
        return NotImplemented

    def __rmul__(self, scalar: int | float) -> ScreenPoint:
        """Handles cases where the scalar is on the left (e.g., 2.0 * point)."""
        return self.__mul__(scalar)

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y, self.z)[index]

    def __len__(self) -> int:
        return 3

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Geometric2D:
    """Base class to provide numpy interoperability and scalar arithmetic for 2D types."""
    val_x:  int | float
    val_y:  int | float

    def __array__(self, dtype=None):
        return np.array([self.val_x, self.val_y], dtype=dtype)

    def __add__(self, other: Geometric2D | np.ndarray | list | tuple | float | int) -> Geometric2D:
        if isinstance(other, Geometric2D):
            return self.__class__(self.val_x + other.val_x, self.val_y + other.val_y)
        elif isinstance(other, (np.ndarray, list, tuple)) and len(other) >= 2:
            return self.__class__(self.val_x + other[0], self.val_y + other[1])
        elif isinstance(other, (int, float)):
            return self.__class__(self.val_x + other, self.val_y + other)
        return NotImplemented

    def __sub__(self, other: Geometric2D | np.ndarray | list | tuple | float | int) -> Geometric2D:
        if isinstance(other, Geometric2D):
            return self.__class__(self.val_x - other.val_x, self.val_y - other.val_y)
        elif isinstance(other, (np.ndarray, list, tuple)) and len(other) >= 2:
            return self.__class__(self.val_x - other[0], self.val_y - other[1])
        elif isinstance(other, (int, float)):
            return self.__class__(self.val_x - other, self.val_y - other)
        return NotImplemented

    def __mul__(self, scalar: int | float) -> Geometric2D:
        if isinstance(scalar, (int, float)):
            return self.__class__(self.val_x * scalar, self.val_y * scalar)
        return NotImplemented

    def __rmul__(self, scalar: int | float) -> Geometric2D:
        return self.__mul__(scalar)

    def __getitem__(self, index: int) -> int | float:
        return (self.val_x, self.val_y)[index]

    def __len__(self) -> int:
        return 2

    def __iter__(self):
        yield self.val_x
        yield self.val_y


@dataclass
class ScreenPoint(Geometric2D):
    # redefine the fields to keep the explicit names x and y
    x: int | float
    y: int | float

    def __init__(self, x: int | float, y: int | float):
        super().__init__(val_x=x, val_y=y)
        self.x = x
        self.y = y

    @classmethod
    def from_array(cls, array: np.ndarray) -> ScreenPoint:
        int_array = np.asarray(array).astype(int)
        return cls(*int_array[:2])

@dataclass
class Extents(Geometric2D):
    width: int | float
    height: int | float

    def __init__(self, width:  int | float, height: int | float):
        super().__init__(val_x=width, val_y=height)
        self.width = width
        self.height = height

    @classmethod
    def from_array(cls, array: np.ndarray) -> Extents:
        int_array = np.asarray(array).astype(int)
        return cls(*int_array[:2])


@dataclass
class Limits(Geometric2D):
    min: float
    max: float

    def __init__(self, min:  float, max: float):
        super().__init__(val_x=min, val_y=max)
        self.min = min
        self.max = max

    @classmethod
    def from_array(cls, array: np.ndarray) -> Limits:
        float_array = np.asarray(array).astype(float)
        return cls(*float_array[:2])


class EyeForm(Enum):
    EYE_ONLY = 1
    MIRROR_ONLY = 2
    EYE_PLUS_MIRROR_ON_PUPIL_AND_IRIS = 3
    EYE_PLUS_MIRROR_EVERYWHERE = 4
    EYE_PLUS_MIRROR_ONLY_ON_PUPIL = 5
    EYE_PLUS_MIRROR_ONLY_ON_IRIS = 6
    EYE_PLUS_MIRROR_ONLY_ON_SCLERA = 7


class KeyboardCode(Enum):
    ESCAPE = 27


@dataclass
class Offset:
    min: float
    max: float
    default: float
    adaptive: bool
