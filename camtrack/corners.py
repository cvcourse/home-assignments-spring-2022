#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

from typing import List

import click
import cv2
import numpy as np
import pims
from numba import jit
from sklearn.neighbors import KDTree

import utils
from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


CORNER_BLOCK_SIZE = 9
CORNER_QUALITY_LEVEL = 0.005
CORNER_MIN_DISTANCE_PX = 20
MAX_CORNERS = 2 ** 31 - 1

PYRAMID_MIN_SIZE_THRESHOLD_PERCENT = 0.1
PYRAMID_SHOW_BLOCKS = False

OPTICAL_FLOW_BLOCK_SIZE = 15
OPTICAL_FLOW_PARAMS = dict(
    winSize=(OPTICAL_FLOW_BLOCK_SIZE, OPTICAL_FLOW_BLOCK_SIZE),
    maxLevel=5,
)


def log(*args, **kwargs):
    print(*args, **kwargs)
    input()


def _to_frame_corners(corners: np.ndarray, ids: np.array = None):
    if ids is None:
        ids = np.array(list(range(len(corners))))

    return FrameCorners(
        ids=ids,
        points=corners,
        sizes=np.full(corners.shape[0], CORNER_BLOCK_SIZE)
    )


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image = frame_sequence[0]
    corners = _get_corners_for_frame(image)
    frame_corners = _to_frame_corners(corners)
    builder.set_corners_at_frame(0, frame_corners)

    prev_image = utils.to_cv_8u(image)
    prev_corners = frame_corners
    prev_ids = frame_corners.ids
    for frame, image in enumerate(frame_sequence[1:], 1):
        image = utils.to_cv_8u(image)

        prev_points = np.float32(prev_corners.points.reshape(-1, 2))

        (optical_flow, is_good, _) = cv2.calcOpticalFlowPyrLK(
            prevImg=prev_image,
            nextImg=image,
            prevPts=prev_points,
            nextPts=None,
            **OPTICAL_FLOW_PARAMS
        )

        (back_optical_flow, _, _) = cv2.calcOpticalFlowPyrLK(
            prevImg=image,
            nextImg=prev_image,
            prevPts=optical_flow,
            nextPts=None,
            **OPTICAL_FLOW_PARAMS
        )

        diff = abs(prev_points - back_optical_flow).reshape(-1, 2).max(-1)
        is_good = diff < 1

        optical_flow = optical_flow.reshape(-1, 2)
        points = []
        ids = []
        for point_index, point in enumerate(optical_flow):
            if is_good[point_index]:
                ids.append(prev_ids[point_index])
                points.append(point)

        points = np.array(points)
        ids = np.array(ids)

        new_corners = _to_frame_corners(points, ids)
        builder.set_corners_at_frame(frame, new_corners)

        prev_image = image
        prev_ids = ids
        prev_corners = new_corners


def _get_corners_for_frame(frame: np.array, use_pyramid=True) -> np.ndarray:
    all_corners: np.ndarray

    if use_pyramid:
        pyramid = _build_pyramid_for_frame(frame)

        raw_corners = []
        pyramid_size = len(pyramid)

        for layer in range(pyramid_size):
            new_corners = _pyramid_find_corners_for_layer(pyramid, layer)
            raw_corners.extend(new_corners)

        raw_corners = np.array(raw_corners)
        filtered_corners = _filter_close_corners(
            np.array(raw_corners),
            CORNER_MIN_DISTANCE_PX
        )

        all_corners = filtered_corners

    else:
        all_corners = _get_corners_for_single_frame(frame)

    return all_corners


def _get_corners_for_single_frame(frame: np.array) -> np.array:
    block_size = CORNER_BLOCK_SIZE

    prepared_frame = _preprocess_image(frame)

    corners = cv2.goodFeaturesToTrack(
        image=prepared_frame,
        maxCorners=MAX_CORNERS,
        qualityLevel=CORNER_QUALITY_LEVEL,
        minDistance=block_size + block_size // 2,
        blockSize=block_size
    )

    return corners.reshape(-1, 2)


def _preprocess_image(frame: np.array) -> np.array:
    prepared_frame = utils.smooth(frame, ksize=7)
    prepared_frame = utils.sharpen(prepared_frame)

    return prepared_frame


def _build_pyramid_for_frame(frame: np.array) -> List[np.array]:
    """
    Build pyramid for the frame.
    Result is list of frames from the smallest image to largest.
    """

    (height, width) = np.shape(frame)
    size_threshold = int(
        min(height, width) * PYRAMID_MIN_SIZE_THRESHOLD_PERCENT
    )

    def frame_is_large_enough(cur_frame: np.array):
        (cur_frame_h, cur_frame_w) = np.shape(cur_frame)
        return cur_frame_h > size_threshold and cur_frame_w > size_threshold

    out = [frame]

    diminished_frame = _diminish_frame_size(frame)
    while frame_is_large_enough(diminished_frame):
        out.append(diminished_frame)
        diminished_frame = _diminish_frame_size(diminished_frame)

    return list(reversed(out))


def _pyramid_find_corners_for_layer(
        pyramid: List[np.array],
        layer: int
) -> np.ndarray:
    pyramid_size = len(pyramid)
    corners = _get_corners_for_single_frame(pyramid[layer])
    corners = _rescale_corners(corners, pyramid_size, layer)
    return corners


def _filter_close_corners(corners: np.ndarray, radius: int) -> np.ndarray:
    """
    Filter out close corners with given radius.
    The result is mean between such corners.

    :param corners: array of points with shape (-1, 2)
    :param radius: threshold in which corners will be joint
    :return: filtered corners, np.array of shape (-1, 2)
    """
    used = np.ones(len(corners), dtype=bool)

    kd_tree = KDTree(corners, metric='manhattan')
    neighbours = kd_tree.query_radius(corners, radius)

    result = []

    for group_index, indices_group in enumerate(neighbours):
        if not used[group_index]:
            continue

        for point_index in indices_group:
            used[point_index] = True

        most_accurate_point_index = indices_group.max()
        result.append(corners[most_accurate_point_index])

    return np.array(result)


def _diminish_frame_size(frame: np.array) -> np.array:
    return cv2.pyrDown(frame)


def _rescale_corners(corners: np.ndarray, pyramid_size: int, layer: int) -> np.ndarray:
    """
    Rescale corners to the original image coordinates.
    Lower layer is lower image.
    """
    coef = _pyramid_coef(pyramid_size, layer)
    return np.float32(corners * coef)


@jit
def _pyramid_coef(pyramid_size: int, layer: int) -> int:
    """
    Return scaling coefficient of the pyramid layer.
    Lower layer is lower image.
    """
    return 2 ** (pyramid_size - layer - 1)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
