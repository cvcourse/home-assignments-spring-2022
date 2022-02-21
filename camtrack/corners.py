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


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    builder.set_corners_at_frame(0, _get_corners_for_frame(image_0))
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        builder.set_corners_at_frame(frame, _get_corners_for_frame(image_1))


def _get_corners_for_frame(frame: np.array) -> FrameCorners:
    block_size = 7

    cv_corners = cv2.goodFeaturesToTrack(
        image=frame,
        maxCorners=2**30,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=block_size
    )

    corners_count = len(cv_corners)

    ids = list(range(0, corners_count))
    points = list(map(lambda i: i[0], cv_corners))
    sizes = list(np.ones(corners_count) * block_size)

    corners = FrameCorners(
        ids=np.array(ids),
        points=np.array(points),
        sizes=np.array(sizes)
    )

    return corners


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
