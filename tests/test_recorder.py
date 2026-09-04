"""The recorder must produce a readable file and never block the pipeline."""

import os
import time

import cv2
import numpy as np

from modules.recorder import RecorderSink, default_output_path


def _frame(width=320, height=180, value=90):
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    cv2.circle(frame, (width // 2, height // 2), 40, (0, 0, 255), -1)
    return frame


def test_default_output_path_is_timestamped(tmp_path):
    first = default_output_path(str(tmp_path))
    assert first.endswith(".mp4")
    assert os.path.dirname(first) == str(tmp_path)


def test_records_a_playable_file(tmp_path):
    path = str(tmp_path / "clip.mp4")
    sink = RecorderSink()
    assert sink.start(path, 320, 180, 30.0), sink.error
    try:
        for index in range(30):
            sink.send(_frame(value=80 + index))
        # Let the encoder thread drain the queue.
        deadline = time.time() + 5
        while sink.frames < 30 and time.time() < deadline:
            time.sleep(0.05)
    finally:
        written = sink.stop()

    assert written == path
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0

    capture = cv2.VideoCapture(path)
    try:
        assert capture.isOpened()
        ok, frame = capture.read()
        assert ok and frame is not None
        assert frame.shape[:2] == (180, 320)
    finally:
        capture.release()


def test_send_is_a_noop_when_stopped():
    sink = RecorderSink()
    # Must not raise even though nothing is open.
    sink.send(_frame())
    assert not sink.is_recording
    assert sink.stop() == ""


def test_odd_dimensions_are_made_even(tmp_path):
    sink = RecorderSink()
    assert sink.start(str(tmp_path / "odd.mp4"), 321, 181, 30.0), sink.error
    try:
        assert sink.resolution_is_even()
    finally:
        sink.stop()


def test_reports_a_failure_for_an_unwritable_path(tmp_path):
    sink = RecorderSink()
    # A directory where a file should be.
    target = tmp_path / "as_a_dir.mp4"
    target.mkdir()
    assert not sink.start(str(target), 320, 180, 30.0)
    assert sink.error
