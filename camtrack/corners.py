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

import os
from dataclasses import dataclass

import click
import cv2
import numpy as np
import pims

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

from typing import List


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
CORNER_QUALITY_LEVEL = 0.05
MAX_CORNERS = 2**31 - 1

PYRAMID_SIZE_THRESHOLD_PERCENT = 0.1
PYRAMID_SHOW_BLOCKS = False


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    builder.set_corners_at_frame(
        0,
        _get_corners_for_frame(image_0)
    )

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        builder.set_corners_at_frame(
            frame,
            _get_corners_for_frame(image_1)
        )


def _get_corners_for_frame(frame: np.array, use_pyramid=True) -> FrameCorners:
    @dataclass
    class CornerWithBlockSize():
        point: tuple
        block_size: int

    if use_pyramid:
        pyramid = _build_pyramid_for_frame(frame)

        all_corners = []
        pyramid_size = len(pyramid)
        for layer in range(pyramid_size):
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
                lambda corner: CornerWithBlockSize(corner, block_size),
                corners
            )
            corners = list(corners)
            all_corners += corners
    else:
        all_corners = map_l(
            lambda x: CornerWithBlockSize(x, CORNER_BLOCK_SIZE),
            _get_corners_for_single_frame(frame)
        )

    corners_count = len(all_corners)

    ids = list(range(corners_count))
    points = map_l(lambda x: x.point, all_corners)
    sizes = map_l(lambda x: x.block_size, all_corners)

    corners = FrameCorners(
        ids=np.array(ids),
        points=np.array(points),
        sizes=np.array(sizes)
    )

    return corners


def _get_corners_for_single_frame(frame: np.array) -> np.array:
    block_size = CORNER_BLOCK_SIZE

    corners = cv2.goodFeaturesToTrack(
        image=frame,
        maxCorners=MAX_CORNERS,
        qualityLevel=CORNER_QUALITY_LEVEL,
        minDistance=block_size + block_size // 2,
        blockSize=block_size
    )
    corners = map(lambda i: i[0], corners)
    corners = list(corners)

    return np.array(corners)


def _build_pyramid_for_frame(frame: np.array) -> List[np.array]:
    """
    Build pyramid for the frame.
    Result is list of frames from the smallest image to largest.
    """

    (height, width) = np.shape(frame)
    size_threshold = int(
        min(height, width) * PYRAMID_SIZE_THRESHOLD_PERCENT
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


def _diminish_frame_size(frame: np.array) -> np.array:
    return cv2.pyrDown(frame)


def _rescale_corner(corner: np.array, pyramid_size: int, layer: int) -> np.array:
    """
    Rescales corner to the original image coordinates.
    Lower layer is lower image.
    """
    coef = _pyramid_coef(pyramid_size, layer)
    return np.array(map_l(lambda x: x * coef, corner))


def _pyramid_coef(pyramid_size: int, layer: int) -> int:
    """
    Return scaling coefficient of the pyramid layer.
    Lower layer is lower image.
    """
    return 2 ** (pyramid_size - layer - 1)


def _dump_pyramid(pyramid: List[np.array]):
    base_path = "out/debug/pyramid"
    os.makedirs(base_path, exist_ok=True)

    for i in range(len(pyramid)):
        # TODO: proper dump
        cv2.imwrite(f"{base_path}/{i}.jpg", pyramid[i])


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


def map_l(f, sequence): return list(map(f, sequence))


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
