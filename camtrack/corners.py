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

from dataclasses import dataclass
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
from utils import map_l


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


def log(*args, **kwargs):
    print(*args, **kwargs)
    input()


@dataclass
class _CornerWithBlockSize:
    point: np.ndarray
    block_size: int


def _to_frame_corners(corners_with_size: List[_CornerWithBlockSize], ids: List[int] = None):
    if ids is None:
        ids = list(range(len(corners_with_size)))

    points = map_l(lambda x: x.point, corners_with_size)
    sizes = map_l(lambda x: x.block_size, corners_with_size)

    return FrameCorners(
        ids=np.array(ids),
        points=np.array(points),
        sizes=np.array(sizes)
    )


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


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image = frame_sequence[0]
    corners_with_size = _get_corners_for_frame(image)
    frame_corners = _to_frame_corners(corners_with_size)
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

        new_corners = FrameCorners(
            ids=ids,
            points=points,
            sizes=prev_corners.sizes
        )

        builder.set_corners_at_frame(frame, new_corners)

        prev_image = image
        prev_ids = ids
        prev_corners = new_corners


def _get_corners_for_frame(frame: np.array, use_pyramid=True) -> List[_CornerWithBlockSize]:
    all_corners: List[_CornerWithBlockSize]

    if use_pyramid:
        pyramid = _build_pyramid_for_frame(frame)

        raw_corners = []
        removed_corners = set()
        pyramid_size = len(pyramid)

        for layer in range(pyramid_size):
            new_corners = _pyramid_find_corners_for_layer(pyramid, layer)
            _pyramid_filter_close_corners(
                raw_corners=raw_corners,
                removed_corners=removed_corners,
                new_corners=new_corners
            )

        all_corners = []
        for (corner_index, corner) in enumerate(raw_corners):
            if corner_index not in removed_corners:
                all_corners.append(corner)

    else:
        all_corners = map_l(
            lambda x: _CornerWithBlockSize(x, CORNER_BLOCK_SIZE),
            _get_corners_for_single_frame(frame)
        )

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
) -> List[_CornerWithBlockSize]:
    pyramid_size = len(pyramid)

    if PYRAMID_SHOW_BLOCKS:
        block_size = CORNER_BLOCK_SIZE * _pyramid_coef(pyramid_size, layer)
    else:
        block_size = CORNER_BLOCK_SIZE

    corners = _get_corners_for_single_frame(pyramid[layer])
    corners = map(
        lambda corner: _rescale_corner(corner, pyramid_size, layer),
        corners
    )
    corners = map(
        lambda corner: _CornerWithBlockSize(corner, block_size),
        corners
    )
    corners = list(corners)

    return corners


def _pyramid_filter_close_corners(raw_corners, removed_corners, new_corners):
    for (corner_index, old_corner) in enumerate(raw_corners):
        if corner_index in removed_corners:
            continue

        for new_corner in new_corners:
            distance = utils.manhattan_distance(
                old_corner.point,
                new_corner.point
            )
            if distance < CORNER_MIN_DISTANCE_PX:
                removed_corners.add(corner_index)

    raw_corners.extend(new_corners)


def _diminish_frame_size(frame: np.array) -> np.array:
    return cv2.pyrDown(frame)


def _rescale_corner(corner: np.array, pyramid_size: int, layer: int) -> np.array:
    """
    Rescales corner to the original image coordinates.
    Lower layer is lower image.
    """
    coef = _pyramid_coef(pyramid_size, layer)
    return np.array(map_l(lambda x: np.float32(x * coef), corner))


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
