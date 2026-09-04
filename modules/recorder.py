"""Record the processed feed to a video file.

A third consumer of the live pipeline, alongside the preview and the virtual
camera. Because the engine fans frames out to any registered sink, recording
costs the swap loop nothing beyond a queue push — the encoder runs on its own
thread and drops frames rather than stalling the pipeline if it falls behind.

Writes with OpenCV's VideoWriter (mp4v), so it does not depend on ffmpeg
being installed. There is no audio: the pipeline only ever sees video frames.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

# A couple of seconds of slack at 30fps. Deep enough to ride out a slow disk
# without letting latency grow unbounded.
_QUEUE_DEPTH = 60


def default_output_path(directory: str, prefix: str = "morphify") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(directory, f"{prefix}-{stamp}.mp4")


class RecorderSink:
    """Frame sink that encodes to an mp4 on a background thread."""

    def __init__(self) -> None:
        self._writer: Optional[cv2.VideoWriter] = None
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_DEPTH)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._size: Tuple[int, int] = (0, 0)
        self._path = ""
        self._error = ""
        self._frames = 0
        self._dropped = 0
        self._started_at = 0.0

    # ── state ────────────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def path(self) -> str:
        return self._path

    @property
    def error(self) -> str:
        return self._error

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    def resolution_is_even(self) -> bool:
        """Both dimensions even, as the mp4v encoder requires."""
        width, height = self._size
        return width % 2 == 0 and height % 2 == 0

    @property
    def elapsed(self) -> float:
        if not self._started_at:
            return 0.0
        return time.time() - self._started_at

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self, path: str, width: int, height: int, fps: float = 30.0) -> bool:
        """Open ``path`` for writing. Returns False and sets ``error``."""
        with self._lock:
            if self._writer is not None:
                return True
            self._error = ""

            if os.path.isdir(path):
                self._error = f"{path} is a folder, not a file."
                return False

            directory = os.path.dirname(path)
            if directory:
                try:
                    os.makedirs(directory, exist_ok=True)
                except OSError as exc:
                    self._error = f"Could not create {directory}: {exc}"
                    return False

            # VideoWriter needs even dimensions and a sane frame rate.
            width = max(2, int(width) - (int(width) % 2))
            height = max(2, int(height) - (int(height) % 2))
            fps = float(max(1.0, min(fps, 120.0)))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

            # isOpened() is not enough: OpenCV reports success for paths it
            # cannot actually write, and the failure would then surface as a
            # recording that silently produced nothing. A successful open
            # creates the file immediately, so check that it is really there.
            if not writer.isOpened() or not os.path.isfile(path):
                try:
                    writer.release()
                except Exception:
                    pass
                self._error = (
                    f"Could not open {os.path.basename(path)} for writing. "
                    "Check the folder is writable."
                )
                return False

            self._writer = writer
            self._size = (width, height)
            self._path = path
            self._frames = 0
            self._dropped = 0
            self._started_at = time.time()

            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

            self._stop.clear()
            self._thread = threading.Thread(
                target=self._pump, name="Recorder", daemon=True)
            self._thread.start()
            return True

    def stop(self) -> str:
        """Finish the file and return its path (empty if nothing was written)."""
        with self._lock:
            if self._writer is None and self._thread is None:
                return ""
            self._stop.set()
            thread, self._thread = self._thread, None

        if thread is not None:
            # Give the encoder a moment to flush what is already queued.
            thread.join(timeout=5.0)

        with self._lock:
            writer, self._writer = self._writer, None
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            path = self._path
            self._path = ""
            self._started_at = 0.0
            return path

    # ── frame path ───────────────────────────────────────────────────────

    def send(self, frame: np.ndarray) -> None:
        """Queue a frame. Never blocks; counts drops instead."""
        if self._writer is None or frame is None:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # Dropping the newest keeps the recording's timing honest: the
            # already-queued frames are older and belong earlier in the file.
            self._dropped += 1

    def _pump(self) -> None:
        width, height = self._size
        while True:
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue

            writer = self._writer
            if writer is None:
                break
            try:
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height),
                                       interpolation=cv2.INTER_AREA)
                writer.write(frame)
                self._frames += 1
            except Exception as exc:
                self._error = f"Recording stopped: {exc}"
                break
