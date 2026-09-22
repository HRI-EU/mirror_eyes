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
from threading import Thread
from time import time

import numpy as np


def threaded(fn: Callable) -> Callable:
    """
    Change function to run in a thread.

    Returns:
    -------
        the wrapped function
    """

    def wrapper(*args, **kwargs) -> Thread:
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


class Color:
    def __init__(self, b: int, g: int, r: int, a: int | None = None) -> None:
        self.update(b=b, g=g, r=r, a=a)

    @property
    def bgr(self):
        return (self.b, self.g, self.r)

    @property
    def bgra(self):
        return (self.b, self.g, self.r, self.a)

    @property
    def rgb(self):
        return (self.r, self.g, self.b)

    @property
    def rgba(self):
        return (self.r, self.g, self.b, self.a)

    def update(self, b: int, g: int, r: int, a: int | None = None) -> None:
        self.b = b
        self.g = g
        self.r = r
        self.a = a


class TransitioningColor(Color):
    def __init__(self, b: int, g: int, r: int, a: int | None = None) -> None:
        self.b = b
        self.g = g
        self.r = r
        self.a = a
        self.last_step = time()
        self.step_duration = 0.0
        self.transition_start = 0.0
        self.target_color = Color(b=b, g=g, r=r, a=a)

    def set_channel_increments(self, max_steps=5) -> None:
        """todo: add alpha support"""
        target_as_array = np.array([self.target_color.b, self.target_color.g, self.target_color.r])
        current_as_array = np.array([self.b, self.g, self.r])
        distances = target_as_array - current_as_array
        # longest = max(np.absolute(distances))
        step_distances = distances // max_steps

        longest = max(np.absolute(step_distances))
        if longest > 0:
            self.step_duration = self.transition_duration / longest
        else:
            self.step_duration = 0.0
        self.last_step = time()
        # increments = np.sign(distances)  #
        self.b_increment = step_distances[0]  # increments[0]
        self.g_increment = step_distances[1]  # increments[1]
        self.r_increment = step_distances[2]  # increments[2]

    def update(self, b: int, g: int, r: int, a: int | None = None, duration: float = 1.0) -> None:
        self.target_color = Color(b=b, g=g, r=r, a=a)
        if duration == 0:
            self.b = b
            self.g = g
            self.r = r
            self.a = a
        if self.target_color.bgr != self.bgr:
            self.transition_duration = duration
            self.set_channel_increments()
            self.transition_start = time()
            # self.color_transition()

    @property
    def transitioning(self):
        return self.bgr != self.target_color.bgr

    def transition_step(self):
        if time() - self.last_step >= self.step_duration:
            # todo: repeat more elegantly for individual channels
            if self.b != self.target_color.b:
                change = int(self.b + self.b_increment)
                if abs(self.target_color.b - change) < abs(self.b_increment):
                    self.b = self.target_color.b
                else:
                    self.b = change
            if self.g != self.target_color.g:
                change = int(self.g + self.g_increment)
                if abs(self.target_color.g - change) < abs(self.g_increment):
                    self.g = self.target_color.g
                else:
                    self.g = change
            if self.r != self.target_color.r:
                change = int(self.r + self.r_increment)
                if abs(self.target_color.r - change) < abs(self.r_increment):
                    self.r = self.target_color.r
                else:
                    self.r = change
            self.last_step = time()

    @threaded
    def color_transition(self):
        while self.transitioning:
            self.transition_step()


if __name__ == "__main__":
    c1 = TransitioningColor(b=100, g=150, r=200)
    c1.update(b=110, g=130, r=300, duration=3)
    prev = c1.bgr
    while c1.transitioning:
        c1.transition_step()
        curr = c1.bgr
        if curr != prev:
            print(c1.bgr)
            prev = curr
    input("press enter to continue with a new color")
    c1.update(b=10, g=200, r=222, duration=3)
    prev = c1.bgr
    while c1.transitioning:
        c1.transition_step()
        curr = c1.bgr
        if curr != prev:
            print(c1.bgr)
            prev = curr
