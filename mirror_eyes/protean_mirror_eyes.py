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
from enum import Enum

# own
from mirror_eyes.mirror_eyes import MirrorEyes


class EyeForm(Enum):
    EYE_ONLY = 1
    MIRROR_ONLY = 2
    EYE_PLUS_MIRROR_ON_PUPIL_AND_IRIS = 3
    EYE_PLUS_MIRROR_EVERYWHERE = 4
    EYE_PLUS_MIRROR_ONLY_ON_PUPIL = 5
    EYE_PLUS_MIRROR_ONLY_ON_IRIS = 6
    EYE_PLUS_MIRROR_ONLY_ON_SCLERA = 7


class ProteanMirrorEyes(MirrorEyes):
    """
    Extension of mirror eyes that can switch between appearances
    with dedicated high level functions. The settings are applied
    equally to all eyes.
    """

    def __init__(self, *args, **kwargs) -> None:
        MirrorEyes.__init__(self, *args, **kwargs)

    def hide_pupil(self) -> None:
        for eye in self.eyes:
            eye.show_pupil = False

    def show_pupil(self) -> None:
        for eye in self.eyes:
            eye.show_pupil = True

    def hide_iris(self) -> None:
        for eye in self.eyes:
            eye.show_iris = False

    def show_iris(self) -> None:
        for eye in self.eyes:
            eye.show_iris = True

    def enable_mirroring(self) -> None:
        for eye in self.eyes:
            eye.enable_mirroring()

    def disable_mirroring(self) -> None:
        for eye in self.eyes:
            eye.disable_mirroring()

    def mirror_on_sclera(self) -> None:
        self.enable_mirroring()
        for eye in self.eyes:
            eye.mirror_on_pupil = True
            eye.mirror_on_iris = True

    def mirror_on_iris_and_pupil(self) -> None:
        self.enable_mirroring()
        for eye in self.eyes:
            eye.mirror_on_pupil = True
            eye.mirror_on_iris = True

    def mirror_only_on_pupil(self) -> None:
        self.enable_mirroring()
        for eye in self.eyes:
            eye.mirror_on_pupil = True
            eye.mirror_on_iris = False

    def mirror_only_on_iris(self) -> None:
        self.enable_mirroring()
        for eye in self.eyes:
            eye.mirror_on_pupil = False
            eye.mirror_on_iris = True

    def mirror_only_on_sclera(self) -> None:
        # not yet implemented because sclera mirroring is still a function of
        # pupil and iris mirroring
        self.enable_mirroring()
        for eye in self.eyes:
            eye.mirror_on_pupil = False
            eye.mirror_on_iris = False
            # eye.mirrro_on_sclera = True

    def set_eye_mode(self, mode: EyeForm) -> None:
        if mode == EyeForm.EYE_ONLY:
            self.show_pupil()
            self.show_iris()
            self.disable_mirroring()
        elif mode == EyeForm.MIRROR_ONLY:
            self.hide_iris()
            self.hide_pupil()
            self.mirror_only_on_sclera()
        elif mode == EyeForm.EYE_PLUS_MIRROR_ON_PUPIL_AND_IRIS:
            self.show_iris()
            self.show_pupil()
            self.mirror_on_iris_and_pupil()
        elif mode == EyeForm.EYE_PLUS_MIRROR_EVERYWHERE:
            self.show_iris()
            self.show_pupil()
            self.mirror_on_sclera()
        elif mode == EyeForm.EYE_PLUS_MIRROR_ONLY_ON_SCLERA:
            self.show_iris()
            self.show_pupil()
            self.mirror_only_on_sclera()
        elif mode == EyeForm.EYE_PLUS_MIRROR_ONLY_ON_IRIS:
            self.show_iris()
            self.show_pupil()
            self.mirror_only_on_iris()
        elif mode == EyeForm.EYE_PLUS_MIRROR_ONLY_ON_PUPIL:
            self.show_iris()
            self.show_pupil()
            self.mirror_only_on_pupil()
        else:
            logging.error(f"{mode} is not a valid eye mode.")
