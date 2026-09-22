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
# Gaze point selection that can keep track
# and switch between multiple faces.
# Tracking is realized through proximity measures
# between subsequent samples.
# Individual thresholds for tracked objects may be
# used.

# pylint: disable=logging-fstring-interpolation, too-few-public-methods, too-many-instance-attributes
# pylint: disable=E0401

from __future__ import annotations

import logging
from random import choice
from time import time

import numpy as np

from mirror_eyes.custom_types import Extents, ScreenPoint
from mirror_eyes_app.attention import FaceFollower


def horizontally_distributed_gaze_targets(
    max_width: int = 1024, max_height: int = 576, target_count: int = 3, norm_distance: float = 0.5
):
    """Creates target_count gaze target objects and returns them in a list
    Their positions are distributed evenly horizontally between 0
    and max_width. Vertically, all are initialized on max_height/2"""

    xstep = max_width // (target_count + 1)
    y = max_height // 2  # target_count
    gaze_targets = []
    for i in range(target_count):
        gaze_targets.append(GazeTarget(identifier=i, position=((i + 1) * xstep, y, norm_distance)))
    logging.info(f"Initial gaze targets horizontally distributed with {xstep} pixel gaps.")
    return gaze_targets


def gaze_targets_at_given_relative_coordinates(
    max_width: int = 1024, max_height: int = 576, relative_coordinates: list[list[float]] | None = None
):
    if relative_coordinates is None:
        relative_coordinates = [[0.3, 0.3], [0.5, 0.5], [0.1, 0.2]]
    h, w = max_height, max_width
    pixel_coordinates = [[int(h * y), int(w * x)] for y, x in relative_coordinates]
    gaze_targets = []
    for id, (y, x) in enumerate(pixel_coordinates):
        gaze_targets.append(GazeTarget(position=(x, y, 0.5), identifier=id))
    logging.info("Initial gaze targets set according to config.")
    return gaze_targets


def face_mapping_from_distance_map(distance_map: np.ndarray):
    """
    Determines what face should be associated with what target_position
    based on a distance map. A distance map is a 2D array with row indices 
    corresponding to targets and column indices corresponding to detected faces.

    It first creates a face index map that is sorted by distance to each target
    (1 row per target).
    In case two targets should have the same face candidate, the one that is closest to the face is assigned to it.

    Args:
        distance_map (nxn array): row_indices = targets, column_indices = detected faces

    Returns:
        A list of closest face indices for each target, ordered by target index

    """
    candidates_in_order = np.argsort(distance_map)
    competition_for_faces = len(candidates_in_order[:, 0]) != len(set(candidates_in_order[:, 0]))
    target_no = len(distance_map)
    loopcount = 0
    while competition_for_faces:
        for i in range(target_no):
            first_choice_i = candidates_in_order[i, 0]
            distance_to_first_choice_i = distance_map[i, first_choice_i]
            for j in range(target_no):
                if i == j:
                    continue
                # if i!=j:
                first_choice_j = candidates_in_order[j, 0]
                distance_to_first_choice_j = distance_map[j, first_choice_j]
                if first_choice_j == first_choice_i:
                    if distance_to_first_choice_j < distance_to_first_choice_i:
                        # np.roll is where the magic happens
                        candidates_in_order[i] = np.roll(candidates_in_order[i], -1)
                        first_choice_i = candidates_in_order[i, 0]
                        distance_to_first_choice_i = distance_map[i, first_choice_i]
                    else:
                        candidates_in_order[j] = np.roll(candidates_in_order[j], -1)
                    competition_for_faces = len(candidates_in_order[:, 0]) != len(set(candidates_in_order[:, 0]))
        loopcount += 1
        if loopcount >= 3:
            logging.error("Could not find a suitable face to target mapping")
            competition_for_faces = False
    return candidates_in_order[:, 0]


class GazeTarget:
    """
    Object with a position property that can be used to keep track
    of gaze targets.

    Args:
      identifier - arbitrary identifier
      position (tuple): coordinates in the form (x, y, distance)
        x and y refer to a position in a camera image
      max_update_distance (number): maximum distance between subsequent
        positions to allow for a position update (currently handled
        externally)
    """

    def __init__(self, identifier: int | None = None, position: tuple | None = None, max_update_distance: int = 120):
        self.identifier = identifier
        self._position = None
        self.position = position
        self.previous_position = position
        self.updated = False
        self.detection = None
        self.max_update_distance = max_update_distance

    @property
    def position(self):
        """returns position"""
        return self._position

    @position.setter
    def position(self, values):
        """Ensures that poi stays within xlim and ylim
        Args:
          values (tuple): coordinates in the form (x, y, distance)
        """
        try:
            x, y, distance = values

        except TypeError as exc:
            raise TypeError("Requires iterable with three items") from exc

        # keep short term memory of previous location
        self.previous_position = self.position
        self._position = (int(x), int(y), distance)


class SelectiveFaceTracker(FaceFollower):
    """
    Gaze point selection class capable of tracking multiple
    faces and following any of these selectively.
    Presently timer-based switching and button-based
    switching between faces are exemplified.

    Args:
        target_count (int): maximum number of faces that should be tracked
            simultaneously.
        no_target_id (int): -1

    Todo: allow for specific setting of start positions
    for the targets. E.g., through config file.

    """

    def __init__(
        self,
        *args,
        target_count: int = 3,
        no_target_id: int = -1,
        name: str = "selective face tracker",
        target_starting_position_estimates: list[tuple] | None = None,
        **kwargs,
    ):
        FaceFollower.__init__(self, *args, name=name, **kwargs)
        self.target_count = target_count
        self.initialize_target_centers(
            target_count=target_count, target_starting_position_estimates=target_starting_position_estimates
        )
        self._target_idx = 0
        self._prev_idx = 0
        # self.no_target_id = no_target_id
        self.no_target_target = GazeTarget(identifier=no_target_id, position=(self.width // 2, self.height // 2, 1))
        self.target_start_time = time()
        self.poi = self.target_centers[self.target_idx].position

    @property
    def target_idx(self):
        """Target index"""
        return self._target_idx

    @target_idx.setter
    def target_idx(self, value):
        """Target index update with validity check"""
        # if value not in range(self.target_count):
        #    logging.warning(f'Invalid index {value}. Target not updated')
        # else:
        self.target_start_time = time()
        self._prev_idx = self._target_idx
        self._target_idx = value

    def look_at_nothing(self):
        self.target_idx = self.no_target_target.identifier

    def target_changed(self):
        """checks if the target_idx has been changed.
        Updates _prev_idx afterwards, making this a
        one time operation per change of target_idx"""
        changed = self._target_idx != self._prev_idx
        if changed:
            logging.debug("target_changed")
            self._prev_idx = self.target_idx
        return changed

    def initialize_target_centers(self, target_count=3, target_starting_position_estimates=None):
        """
        Create n=target_count GazeTarget objects either at given
        starting positions or at automatically determined positions.
        The outcome is stored in self.target_centers

        Target index convention:
            from left to right and from top to bottom

        Args:
            target_count (int): number of GazeTargets
            target_starting_position_estimates (list): list of coordinate tuples

        """
        if target_count is None:
            target_count = self.target_count
        else:
            self.target_count = target_count

        # set best guess default if nothing is provided
        if target_starting_position_estimates is None:
            self.target_centers = horizontally_distributed_gaze_targets(
                max_width=self.width, max_height=self.height, target_count=target_count
            )
        else:
            self.target_centers = gaze_targets_at_given_relative_coordinates(
                max_width=self.width, max_height=self.height, relative_coordinates=target_starting_position_estimates
            )
            target_count = len(self.target_centers)
        logging.info(f"Initialized {target_count} targets")
        for i, target_center in enumerate(self.target_centers):
            logging.debug(f"{i + 1} at {target_center.position}")

    def distance_map(self, face_positions):
        """
        Calculate the distanes between all target locations and detected faces.

        Args:
          face_positions (list): list of coordinate tuples

        Returns:
        2D np.array with targets as rows and faces as columns"""
        distance_map = np.zeros([self.target_count, len(face_positions)])
        for i, target_center in enumerate(self.target_centers):
            for j, face_position in enumerate(face_positions):
                distance_map[i][j] = np.linalg.norm(np.array(face_position[:2]) - np.array(target_center.position[:2]))
        return distance_map

    def update_target_centers_from_detections(self, face_detections):
        """In the presence of detected faces, the target centers
        may be updated for faces that are sufficiently close to an
        existing target center."""
        if not face_detections:
            return
        face_positions = [self.detection_to_poi(face_detection) for face_detection in face_detections]
        # inventing missing faces as far away points
        missing_faces = 0
        while len(face_positions) < len(self.target_centers):
            face_positions.append((9000 + missing_faces,) * 3)
            missing_faces += 1
        distances = self.distance_map(face_positions)
        face_selections = face_mapping_from_distance_map(distances)
        for i, target_center in enumerate(self.target_centers):
            if distances[i, face_selections[i]] < target_center.max_update_distance:
                self.target_centers[i].position = face_positions[face_selections[i]]
                self.target_centers[i].updated = True
                try:
                    self.target_centers[i].detection = face_detections[face_selections[i]]
                except IndexError:
                    self.target_centers[i].detection = None
            else:
                pass

    def count_identified_participants(self):
        n = 0
        for tc in self.target_centers:
            if tc.updated:
                n += 1
        return n

    def update_target_centers_from_image(self, img):
        """Combines face detection and
        update_target_centers_from_detections"""
        results = self.face_detection.process(img)
        self.update_target_centers_from_detections(face_detections=results.detections)

    def relative_to_absolute_bounding_box(self, xmin, ymin, rel_width=0.25, rel_height=0.25, padding=250):
        width = int(rel_width * self.width) + padding * 2
        height = int(rel_height * self.height) + padding * 2
        top_left = ScreenPoint(x=int(xmin * self.width) - padding, y=int(ymin * self.height) - padding)
        return top_left, Extents(width=width, height=height)

    def update_poi(self):
        """POI update based on the current target position"""
        self.poi = self.target_centers[self.target_idx].position

    def update_poi_bounding_box(self):
        if self.target_centers[self.target_idx].detection is not None:
            detection = self.target_centers[self.target_idx].detection
            top_left, extents = self.relative_to_absolute_bounding_box(
                xmin=detection.location_data.relative_bounding_box.xmin,
                ymin=detection.location_data.relative_bounding_box.ymin,
                rel_width=detection.location_data.relative_bounding_box.width,
                rel_height=detection.location_data.relative_bounding_box.height,
            )
        else:
            top_left, extents = self.relative_to_absolute_bounding_box(
                xmin=self.target_centers[self.target_idx].position[0],
                ymin=self.target_centers[self.target_idx].position[1],
            )

        self.poi_bounding_box.update(top_left=top_left, extents=extents)

    def update_target_index_if_expired(self, max_time=3):
        """For automatic switching between targets after some
        time has passed"""
        if self.target_count < 2 or max_time < 0:
            return
        if time() - self.target_start_time > max_time:
            pool = list(range(self.target_count))
            pool.remove(self.target_idx)
            self.target_idx = choice(pool)
            logging.debug(f"updated target index to {self.target_idx}")
            logging.debug(f"new target center {self.target_centers[self.target_idx].position}")

    def update(self, img):
        """Patch update to include multi-face tracking and switching between
        POIs
        Args:
            img (cv2 image) - current frame from camera"""
        if self.target_idx == self.no_target_target.identifier:
            self.poi = self.no_target_target.position
        else:
            self.update_target_centers_from_image(img)
            self.update_target_index_if_expired(-1)
            self.update_poi()
            self.update_poi_bounding_box()
