#!/usr/bin/env python
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
# Simple gaze point selection module. 
# Can be used by mirror eyes to select 
# a point of interest on a given image

from __future__ import annotations

import logging
import random
from collections import deque
from time import time

import mediapipe as mp
import numpy as np

# own
from mirror_eyes.custom_types import Extents, ScreenPoint
from mirror_eyes.tools.bounding_box import SmoothBoundingBox2D
from mirror_eyes.tools.converting import normalize, restrict_to_range

# pylint: disable=logging-fstring-interpolation,E0401


def coordinate(values: tuple[int, int, int]):
    try:
        x, y, z = values
        return (int(x), int(y), int(z))
    except TypeError as exc:
        raise TypeError("Requires iterable with three items") from exc


def distance_from_size(
    detection, 
    img_height: int, 
    normalized: bool = True, 
    maxdistance: int = 500, 
    mindistance: int = 0,
    real_obj_height = 190,  # in mm (e.g., 190 for avg head size) 
    sensor_height = 1.524,  # derived as 96 / 1440 inch to mm
    focal_length = 1, # = 100mm
):
    """Distance calculation based on approximate object size and camera properties
    Returns approximate distance in mm or optionally
    normalized.

    FOCAL LENGTH AND SENSOR HEIGHT MAY REQUIRE ADJUSTMENTS
    DEPENDING ON THE CAMERA    
    """
    obj_pixel_height = detection.location_data.relative_bounding_box.height * img_height
    obj_height_on_sensor = (sensor_height * obj_pixel_height) / img_height
    distance_to_object = (real_obj_height * focal_length) / obj_height_on_sensor
    if normalized:
        distance = normalize(
            val_source=restrict_to_range(distance_to_object, lower=mindistance, upper=maxdistance),
            min_source=mindistance,
            max_source=maxdistance,
            min_target=0.0,
            max_target=1.0,
        )
    else:
        distance = int(distance_to_object)
    return distance


class Attention:
    """
    A gaze point selector (Attention Allocator) selects a
    point to attend/gaze at, given an input image.
    """

    def __init__(
        self,
        name: str = "Attention",
        width: int = 360,
        height: int = 360,
        mirror_opacity: float = 0.5,
        xlim: tuple[int, int] | None = None,
        ylim: tuple[int, int] | None = None,
    ):
        """
        Args:
            xlim (int tuple): smallest and largest possible x coordinate
            ylim (int tuple): smallest and largest possible y coordinate
    
        """
        self.name = name
        if xlim is None:
            self.xlim = (0, 360)
        else:
            self.xlim = xlim
        if ylim is None:
            self.ylim = (0, 360)
        else:
            self.ylim = ylim
        self.mirror_opacity = mirror_opacity
        self.width = width
        self.height = height
        self._poi = coordinate((width / 2, height / 2, 200))
        self.prev_poi = self._poi
        self.poi_bounding_box = SmoothBoundingBox2D(
            top_left=ScreenPoint(x=0, y=0), 
            extents=Extents(width=width, height=height), 
            hysteresis=6
        )
        self.stare = False
        logging.info(f"Initialization of {self.name}")

    @property
    def poi(self):
        return self._poi

    @poi.setter
    def poi(self, values):
        """Ensures that poi stays within xlim and ylim"""
        try:
            x, y, distance = values
        except TypeError as exc:
            raise TypeError("Requires iterable with two items") from exc
        poix = min(max(self.xlim[0], x), self.xlim[1])
        poiy = min(max(self.ylim[0], y), self.ylim[1])

        # keep short term memory of previous location
        self.prev_poi = self.poi
        self._poi = (int(poix), int(poiy), distance)

    def center_poi(self):
        self.poi = (int(self.width / 2), int(self.height / 2))

    def img_to_poi(self, img):
        """Given an image, select a position to attend
        assumes that the image is an indexable np.array
        like object.
        --- Substitute with actual poi selection ---
        --- Currently only the image center is returned ---"""
        # placeholder: select the center of the image
        (h, w) = img.shape[:2]
        self.poi = (h / 2, w / 2)

    def apply_scale_offset(self, img_scale_based_offset):
        # required for compatibility with roi scaling
        self.poi_bounding_box.apply_offset(img_scale_based_offset) 

    def update(self, img):
        """update self.poi"""
        self.img_to_poi(img)

    def shutdown(self):
        pass


class FaceFollower(Attention):
    """
    Extends Attention to include
    - face tracking
    - distance estimation
    - opacity control
    - change detection
    - target priorization (based on proximity)
    """

    def __init__(
        self,
        *args,
        maxdistance: int = 500,
        mindistance: int = 0,
        name: str = "face follower",
        memory_span: int = 2,
        max_mirror_time: float = 3.0,
        minimum_opacity: float = 0.3,
        maximum_opacity: float = 1,
        **kwargs,
    ):
        Attention.__init__(self, *args, name=name, **kwargs)
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,  # 1 works better on larger distances than 0
        )
        self.has_detection = False
        self.detection_count = 0
        self.detection_count_history_length = 10
        self.detection_count_history = deque(maxlen=self.detection_count_history_length)
        self.detection_change = True
        self.detection = None
        self.maxdistance = maxdistance
        self.mindistance = mindistance
        self.minimum_opacity = minimum_opacity
        self.maximum_opacity = maximum_opacity

        self.max_mirror_time = max_mirror_time
        self.reset_mirror_time()
        self.allow_mirror = True
        self._focus_index = 0
        self.mirror_on_pupil = True
        self.mirror_on_iris = True
        self.memory_span = memory_span
        self.last_seen_time = time()

    @property
    def focus_index(self):
        return self._focus_index

    @focus_index.setter
    def focus_index(self, value):
        if isinstance(value, int):
            if self._focus_index != value:
                # logging.debug(f'changing focus index from {self._focus_index} to {value}')
                self._focus_index = value
        else:
            logging.error(f"{value} is not a valid index")

    def shutdown(self):
        self.face_detection.close()
        logging.info("face detection exit")

    def detection_to_absolute_bounding_box(self, detection, padding=250):
        width = int(detection.location_data.relative_bounding_box.width * self.width) + padding * 2
        height = int(detection.location_data.relative_bounding_box.height * self.height) + padding * 2
        top_left = ScreenPoint(
            x=int(detection.location_data.relative_bounding_box.xmin * self.width) - padding,
            y=int(detection.location_data.relative_bounding_box.ymin * self.height) - padding,
        )
        return top_left, Extents(width=width, height=height)
        # return BoundingBox2D(top_left=top_left, width=width, height=height)

    def detection_center(self, detection):
        """Given a detection, return its center"""
        # print(detection.location_data)
        centerx = int(
            (
                detection.location_data.relative_bounding_box.xmin
                + detection.location_data.relative_bounding_box.width / 2
            )
            * self.width
        )
        centery = int(
            (
                detection.location_data.relative_bounding_box.ymin
                + detection.location_data.relative_bounding_box.height / 2
            )
            * self.height
        )
        return (centerx, centery)

    def reset_mirror_time(self):
        self.mirror_start_time = time()

    def check_mirror_time(self):
        if self.max_mirror_time >= 0:
            if time() - self.mirror_start_time > self.max_mirror_time:
                self.mirror_opacity = self.minimum_opacity
                self.mirror_on_pupil = True
                self.mirror_on_iris = True
                self.allow_mirror = False
            else:
                # mirror fading handeled by eye model hysteresis
                if not self.allow_mirror:
                    self.allow_mirror = True
                    self.mirror_on_pupil = False
                    self.mirror_on_iris = False
                    self.mirror_opacity = self.maximum_opacity

    def select_tracked_face(self, detections, face_number=1):
        """
        From a set of mediapipe face detections select the one that
        has its location closest to the previously selected point.
        Selection is done by setting a focus index.

        Initial selection:
        - if there is just one detection, select it
        - if there are two detections, select the one that is closer to the
          camera
        If a selection has been made in the previous iteration
        - take the face that is closer to the previous iteration's location
          than the other

        """
        # 1 = closest to previously selected
        # 2 = 2nd closest to previously selected
        # 3 = most distant to previously selected
        remembers_face = time() - self.last_seen_time < self.memory_span
        if self.detection_count > 0:
            secondidx = 1
            thirdidx = 1
            # Note: The POI will only be updated if focus_index > 0
            firstidx = -1  # 0 # consider defaulting to -1 to avoid model
            # if a target has been tracked
            # continue to do so preferably by checking what detection
            # lies closest to the previous
            if self.focus_index != -1:
                self.last_seen_time = time()
                min_distance = 100  # above this, the identity is considered different
                max_distance = 0
                for idx, detection in enumerate(detections):
                    detection_loc = self.detection_center(detection)
                    distance_to_prev = np.linalg.norm(np.array(detection_loc) - np.array(self.poi[:2]))  
                    if distance_to_prev < min_distance:
                        min_distance = distance_to_prev
                        secondidx = firstidx
                        firstidx = idx
                    # determine most distant face to previous point
                    if distance_to_prev > max_distance:
                        max_distance = distance_to_prev
                        thirdidx = idx
                    # if nothing is close enough, default to -1 (no suitable target)
            # if a target has not been tracked
            elif self.focus_index == -1 and not remembers_face:
                firstidx = secondidx = thirdidx = -1
                logging.debug("changing target")
                self.last_seen_time = time()
                # select a new target (e.g., closest face)
                maxsize = 0
                # score = 0
                for idx, detection in enumerate(detections):
                    detection_size = detection.location_data.relative_bounding_box.height
                    # if the size of the new detection is larger than the current
                    # maximum size by a given fraction (0-1), select the other
                    if detection_size > maxsize:
                        maxsize = detection_size
                        secondidx = firstidx  # the old max becomes second
                        # score = detection.score[0]
                        firstidx = idx
                        thirdidx = idx

            if face_number == 1:
                self.focus_index = firstidx
            elif face_number == 2:
                self.focus_index = secondidx
            elif face_number == 3:
                self.focus_index = thirdidx

        # this can be reached due to the adjustable sensitivity
        # of detection change detection / detection counting
        else:
            if not remembers_face:
                logging.debug("Forgot tracked face")
                # indicate readyness for a new selection
                self.focus_index = -1

    def select_face(self, detections, face_number=1):
        if self.detection_count > 0:
            maxsize = 0
            maxidx = 0
            score = 0
            secondidx = 1
            # face selection based on face size (assuming a larger face is
            # closer). When two faces are near to each other the one with
            # the smaller index is selected
            for idx, detection in enumerate(detections):
                detection_size = detection.location_data.relative_bounding_box.height
                # if the size of the new detection is larger than the current
                # maximum size by a given fraction (0-1), select the other
                if detection_size - maxsize > 0.2:
                    maxsize = detection_size
                    secondidx = maxidx  # the old max becomes second
                    score = detection.score[0]
                    maxidx = idx
            if face_number == 1:
                # select largest face
                self.focus_index = maxidx
            elif face_number == 2:
                # select 2nd largest face
                self.focus_index = secondidx
            logging.debug(f"score: {score}")
        else:
            self.focus_index = 0

    def select_target(self, detections):
        if not self.stare:
            self.select_tracked_face(detections, face_number=1)

    def detect_detection_changes(self, detections):
        """Detection changes are determined by changes
        in the number of detections.
        The sensitivity to detection changes depends on
        the detection_count_history. The longer this deque,
        the less influence comes from a single observation
        (e.g., due to brief detection drops)."""
        try:
            current_count = len(detections)
        except TypeError:
            current_count = 0
        self.detection_count_history.append(current_count)
        # if len(detections) != self.detection_count:
        if self.detection_count != round(np.mean(self.detection_count_history)):
            self.detection_count = current_count
            self.detection_count_history.extend(self.detection_count_history_length * [current_count])
            logging.debug(f"updated detection count, now {self.detection_count}")
            self.reset_mirror_time()
            return True
        else:
            return False

    def detection_to_poi(self, detection):
        centerx, centery = self.detection_center(detection)
        distance = distance_from_size(
            detection=detection, img_height=self.height, maxdistance=self.maxdistance, mindistance=self.mindistance
        )
        return (centerx, centery, distance)

    def random_location(self, minx=0.2, maxx=0.8, miny=0.2, maxy=0.8, mind=1, maxd=1):
        """return a random coordinate on the objects space"""
        x = random.uniform(minx, maxx) * self.width
        y = random.uniform(miny, maxy) * self.height
        distance = random.uniform(mind, maxd) if mind != maxd else mind
        return (x, y, distance)

    def img_to_poi(self, img):
        """
        Generate POI for current iteration based on image data.
        For mirroring purposes, this method may also change
        mirroring parameters such as opacity
        """
        results = self.face_detection.process(img)

        # conversion required for this detection method
        self.detection_change = self.detect_detection_changes(results.detections)
        if results.detections:
            self.has_detection = True
            self.select_target(results.detections)
            # a negative focus index can be used to skip tracking
            # e.g., for inserting pois by other means.
            if self.focus_index >= 0:
                self.detection = results.detections[self.focus_index]
                self.poi = self.detection_to_poi(self.detection)
                top_left, extents = self.detection_to_absolute_bounding_box(self.detection)
                self.poi_bounding_box.update(top_left=top_left, extents=extents)
        else:
            self.has_detection = False
        self.check_mirror_time()
