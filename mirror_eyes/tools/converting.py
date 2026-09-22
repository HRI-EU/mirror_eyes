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

import math
import random


def dpi_to_dpcm(dpi):
    """dots per inch to dots per centimeter conversion"""
    return 0.3937008 * dpi


def apply_noise(value, min_noise=-0.2, max_noise=0.2):
    """random perturbation to simulate microsaccades"""
    return value + random.uniform(min_noise, max_noise)


def restrict_to_range(number, lower, upper):
    """keep number within [lower, upper] range"""
    if upper < lower:
        # switch order if necessary
        lower, upper = upper, lower
    return max(min(number, upper), lower)


# general support functions
def inrange(number, lower, upper):
    """check if a given number lies between two values"""
    if lower < upper:
        return lower <= number <= upper
    else:
        return upper <= number <= lower


def remap_coordinates(val_source, min_source=0.0, max_source=10.0, min_target=0, max_target=1280):
    """Remap/scale from one coordinate system to another.
    Can be used to convert a metric value to a screen/canvas pixel value
    Adds a range restriction to regular normalization."""
    return int(
        restrict_to_range(
            normalize(
                val_source=val_source,
                min_source=min_source,
                max_source=max_source,
                min_target=min_target,
                max_target=max_target,
            ),
            min_target,
            max_target,
        )
    )


def normalize(val_source, min_source=0.0, max_source=1.0, min_target=10.0, max_target=0.0):
    """linear normalization from a source (orig) to a target range.
    Can also be used to return from a [0.,1.] range to a time range or similar.

    Args:
        val_source (float): value in range [min_source, max_source]
        min_source (float): value at which val_source would be minimal
        max_source (float): value at which val_source would be maximal
        min_target (float): value at which val_target would be minimal
        max_target (float): value at which val_target would be maximal

    Returns:
        value in range [min_target, max_target]
    """
    if max_source == min_source:
        return min_target
    val_target = (max_target - min_target) / (max_source - min_source) * (val_source - max_source) + max_target
    # if not inrange(val_source, min_source, max_source):
    #    logging.warning('input %f outside of range [%f, %f]', val_source, min_source, max_source)
    return val_target


def euclidean_distance(p1, p2):
    #dx = p1[0] - p2[0]
    #dy = p1[1] - p2[1]
    #return math.hypot(dx, dy)
    return math.dist(p1, p2)  # arbitrary dimensions
