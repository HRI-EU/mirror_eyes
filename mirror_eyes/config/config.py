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
#
# Usage:
# >>> from mirror_eyes.config import get_config
# >>> config = get_config('external_eye_control')
#

from __future__ import annotations

import importlib
import logging
import sys

from mirror_eyes.config.config_base import (
    IRIS_COLORS,
    CameraConfig,
    DisplayConfig,
    EyeConfig,
    IrisConfig,
    MirrorConfig,
    MirrorEyesConfig,
    OpacityConfig,
    PupilConfig,
)
from mirror_eyes.config.config_external_control import config_external_eye_control
from mirror_eyes.config.config_local_control import config_local_eye_control

__all__ = [
    "IRIS_COLORS",
    "CameraConfig",
    "DisplayConfig",
    "EyeConfig",
    "IrisConfig",
    "MirrorConfig",
    "OpacityConfig",
    "PupilConfig",
]

_configs = [
    config_local_eye_control, 
    config_external_eye_control, 
]


class Config:
    _instance = None
    _active_config: MirrorEyesConfig | None = None

    def __new__(cls) -> MirrorEyesConfig:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Put any initialization here.
        return cls._instance

    @staticmethod
    def add_config(config: MirrorEyesConfig) -> None:
        _configs.append(config)

    def _get_config(self, name: str) -> MirrorEyesConfig:
        if name and ".py" in str(name):
            logging.info("load custom config from file: %s", name)
            return self._load_config_module(name)
        if name:
            logging.info("load custom config from name: %s", name)
            return self._get_config_by_name(name)
        return self._get_config_by_name("local_eye_control")

    @staticmethod
    def _get_config_by_name(name: str) -> MirrorEyesConfig:
        for entry in _configs:
            if entry.name == name:
                return entry
        raise ValueError

    def set_config(self, name: str) -> None:
        self._active_config = self._get_config(name)

    def get_config(self) -> MirrorEyesConfig:
        if not self._active_config:
            self.set_config("local_eye_control")
        if not self._active_config:
            raise ValueError
        return self._active_config

    @staticmethod
    def _load_config_module(source: str) -> MirrorEyesConfig:
        spec = importlib.util.spec_from_file_location("custom_config", source)
        if not spec:
            logging.info("can't load custom config as spec")
            sys.exit(1)
        module = importlib.util.module_from_spec(spec)
        sys.modules["custom_config"] = module
        if not module:
            logging.info("problem with generating module from spec")
            sys.exit(1)
        if not spec.loader:
            logging.info("spec has no loader")
            sys.exit(1)
        spec.loader.exec_module(module)
        config: MirrorEyesConfig = module.custom_config
        return config
