"""Threaded live face-swap pipeline, decoupled from any UI toolkit.

Three stages run concurrently so a slow one never stalls the others:

    camera ──▶ [capture thread] ──▶ raw queue ──▶ [processing thread] ──▶ sinks

Both queues are depth-limited and drop the oldest frame on overflow: for a
live feed a fresh frame is always worth more than a complete backlog.

The processing stage fans its output out to any number of registered sinks —
the on-screen preview and the virtual camera are two independent consumers of
the same frame, and neither can back-pressure the swap loop.  Adding a sink is
how you tap the processed feed for anything new (a recorder, a stream, a
second preview).

This module deliberately has no Qt imports.  The UI drives it; it does not
know the UI exists.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import List, Optional, Protocol

import cv2
import numpy as np

import modules.globals
from modules.face_analyser import (
    detect_many_faces_fast,
    detect_one_face_fast,
    ensure_landmarks,
)
from modules.face_identity import build_from_path
from modules.gpu_processing import gpu_flip
from modules import live_portrait, takeover
from modules.processors.frame.core import get_frame_processors_modules
from modules.video_capture import VideoCapturer

# One in flight, one being worked on.  Deeper queues only buy latency.
_QUEUE_DEPTH = 2


def source_token(path: Optional[str]) -> Optional[tuple]:
    """Identity of a source image: path plus size and modification time.

    Used to decide whether the cached source face is still valid. Comparing
    paths alone is not enough because files get replaced in place, and
    hashing the contents every frame would be wasteful.
    """
    if not path:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return (path, None, None)
    return (path, stat.st_size, stat.st_mtime_ns)


class FrameSink(Protocol):
    """Anything that can receive processed BGR frames."""

    def send(self, frame: np.ndarray) -> None: ...


class LatestFrameSink:
    """Sink that keeps only the most recent frame, for pull-based consumers.

    The UI paints on a timer rather than on frame arrival, so it wants
    "whatever is current" instead of a stream it has to keep up with.
    """

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._generation = 0

    def send(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._generation += 1

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame

    def take(self) -> Optional[np.ndarray]:
        """Return the current frame only if it changed since the last take."""
        with self._lock:
            if self._generation == 0:
                return None
            self._generation = 0
            return self._frame


class _CaptureWorker(threading.Thread):
    """Reads frames from the camera into a bounded queue. Drops on overflow."""

    def __init__(self, cap, capture_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(name="LiveCapture", daemon=True)
        self._cap = cap
        self._queue = capture_queue
        self._stop_flag = stop_event

    def run(self) -> None:
        while not self._stop_flag.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._stop_flag.set()
                break
            _put_latest(self._queue, frame)


class _ProcessingWorker(threading.Thread):
    """Pulls raw frames, runs detect/swap/enhance, pushes to every sink."""

    def __init__(self, engine: "LiveEngine", capture_queue, stop_event,
                 camera_fps: float):
        super().__init__(name="LiveProcessing", daemon=True)
        self._engine = engine
        self._cq = capture_queue
        self._stop_flag = stop_event
        self._fps = camera_fps

    def run(self) -> None:
        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        source_image = None
        last_source_token = None
        prev_time = time.time()
        fps_update_interval = 0.5
        frame_count = 0
        fps = 0.0
        det_count = 0
        cached_target_face = None
        cached_many_faces = None
        # Re-detect a fraction of the frames; tracking between detections is
        # what keeps the swap real-time. The ratio is a performance dial.
        det_ratio = getattr(modules.globals, "detect_interval_ratio", 0.08)
        det_interval = max(1, round(self._fps * det_ratio))
        compositor = takeover.Compositor()
        animator = live_portrait.PortraitAnimator()
        animator_source = None

        while not self._stop_flag.is_set():
            try:
                frame = self._cq.get(timeout=0.05)
            except queue.Empty:
                continue

            if self._engine.reload_processors_requested():
                frame_processors = get_frame_processors_modules(
                    modules.globals.frame_processors
                )
                det_ratio = getattr(modules.globals, "detect_interval_ratio", 0.08)
                det_interval = max(1, round(self._fps * det_ratio))

            temp_frame = frame
            if modules.globals.live_mirror:
                temp_frame = gpu_flip(temp_frame, 1)

            # Panic switch: keep the stream alive but stop swapping. Placed
            # before any model work so it also drops the GPU cost instantly.
            if modules.globals.bypass_swap:
                self._engine.last_original = temp_frame
                self._engine.dispatch(temp_frame)
                continue

            # The split view compares before and after, so the untouched
            # frame has to be kept before the processors write into it.
            if modules.globals.split_view:
                self._engine.last_original = temp_frame.copy()

            # Portrait animation is an alternative to swapping, not an
            # addition: it regenerates the head rather than painting over it,
            # so the swap chain is skipped entirely.
            if modules.globals.portrait_enabled and not modules.globals.map_faces:
                det_count += 1
                if det_count % det_interval == 0 or cached_target_face is None:
                    cached_target_face = detect_one_face_fast(temp_frame)

                prepared = live_portrait.cache.get(modules.globals.source_path)
                if prepared is not None and prepared is not animator_source:
                    try:
                        animator.set_source(prepared)
                        animator_source = prepared
                    except Exception as exc:
                        self._engine.note_error(f"Portrait: {exc}")

                if animator.ready and cached_target_face is not None:
                    try:
                        animated = animator.animate(
                            temp_frame, getattr(cached_target_face, "kps", None))
                        if animated is not None:
                            temp_frame = animated
                    except Exception as exc:
                        self._engine.note_error(f"Portrait: {exc}")

                now = time.time()
                frame_count += 1
                if now - prev_time >= fps_update_interval:
                    fps = frame_count / (now - prev_time)
                    frame_count = 0
                    prev_time = now
                    self._engine._fps = fps
                if modules.globals.show_fps:
                    cv2.putText(
                        temp_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    )

                self._engine.dispatch(temp_frame)
                continue

            if not modules.globals.map_faces:
                # Keyed on the file's identity, not just its path: the
                # random-face button rewrites one temp file in place, so a
                # path-only check meant every face after the first was
                # ignored while the UI happily showed the new thumbnail.
                token = source_token(modules.globals.source_path)
                if token is not None and token != last_source_token:
                    last_source_token = token
                    source_image = build_from_path(
                        modules.globals.source_path,
                        blend=modules.globals.blend_identity)

                det_count += 1
                if det_count % det_interval == 0:
                    if modules.globals.many_faces:
                        cached_target_face = None
                        cached_many_faces = detect_many_faces_fast(temp_frame)
                    else:
                        cached_target_face = detect_one_face_fast(temp_frame)
                        cached_many_faces = None

                cached_faces = None
                if cached_many_faces:
                    cached_faces = cached_many_faces
                elif cached_target_face is not None:
                    cached_faces = [cached_target_face]

                # Fast detection skips the 2d106 landmark model, but the mouth
                # mask needs it. Attach landmarks on demand (computed once per
                # detection cycle — the helper no-ops if already present).
                if modules.globals.mouth_mask and cached_faces:
                    ensure_landmarks(temp_frame, cached_faces)

                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN256":
                        if modules.globals.fp_ui.get("face_enhancer_gpen256", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN512":
                        if modules.globals.fp_ui.get("face_enhancer_gpen512", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-SWAPPER":
                        swapped_bboxes = []
                        if modules.globals.many_faces and cached_many_faces:
                            result = temp_frame.copy()
                            for t_face in cached_many_faces:
                                result = fp.swap_face(source_image, t_face, result)
                                if hasattr(t_face, "bbox") and t_face.bbox is not None:
                                    swapped_bboxes.append(t_face.bbox.astype(int))
                            temp_frame = result
                        elif cached_target_face is not None:
                            temp_frame = fp.swap_face(
                                source_image, cached_target_face, temp_frame
                            )
                            if (
                                hasattr(cached_target_face, "bbox")
                                and cached_target_face.bbox is not None
                            ):
                                swapped_bboxes.append(cached_target_face.bbox.astype(int))
                        temp_frame = fp.apply_post_processing(temp_frame, swapped_bboxes)
                    else:
                        temp_frame = fp.process_frame(source_image, temp_frame)
            else:
                modules.globals.target_path = None
                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame_v2(temp_frame)
                    elif fp.NAME in ("DLC.FACE-ENHANCER-GPEN256", "DLC.FACE-ENHANCER-GPEN512"):
                        fp_key = fp.NAME.split(".")[-1].lower().replace("-", "_")
                        if modules.globals.fp_ui.get(fp_key, False):
                            temp_frame = fp.process_frame_v2(temp_frame)
                    else:
                        temp_frame = fp.process_frame_v2(temp_frame)

            # Full Takeover rides on top of whatever the processors did:
            # it needs the swapped face already in place, and the target's
            # keypoints to know where the head is.
            if modules.globals.takeover_enabled and not modules.globals.map_faces:
                appearance = takeover.cache.get(modules.globals.source_path)
                head = cached_target_face
                if appearance is not None and head is not None:
                    try:
                        temp_frame = compositor.process(
                            temp_frame, appearance,
                            keypoints=getattr(head, "kps", None),
                            hair=modules.globals.takeover_hair,
                            skin=modules.globals.takeover_skin,
                            background=modules.globals.takeover_background,
                            skin_strength=modules.globals.takeover_skin_strength,
                            hair_volume=modules.globals.takeover_hair_volume,
                            face_box=getattr(head, "bbox", None),
                        )
                    except Exception as exc:
                        self._engine.note_error(f"Takeover: {exc}")

            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time
                self._engine._fps = fps

            if modules.globals.show_fps:
                cv2.putText(
                    temp_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )

            self._engine.dispatch(temp_frame)


def _put_latest(q: queue.Queue, item) -> None:
    """Enqueue, evicting the oldest item if the queue is full."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


class LiveEngine:
    """Owns the camera and the capture/processing threads for live mode."""

    def __init__(self) -> None:
        self._cap: Optional[VideoCapturer] = None
        self._capture_worker: Optional[_CaptureWorker] = None
        self._processing_worker: Optional[_ProcessingWorker] = None
        self._stop_event = threading.Event()
        self._sinks: List[FrameSink] = []
        self._sink_lock = threading.Lock()
        self._fps = 0.0
        self._error = ""
        self._reload_processors = threading.Event()
        # Most recent pre-swap frame, published only while the split view or
        # bypass is on so the normal path does no extra copying.
        self.last_original: Optional[np.ndarray] = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self, camera_index: int, width: int = 960, height: int = 540,
              fps: int = 60) -> bool:
        if self.is_running:
            return True

        self._error = ""
        try:
            cap = VideoCapturer(camera_index)
        except Exception as exc:
            self._error = f"Could not open camera: {exc}"
            return False

        if not cap.start(width, height, fps):
            self._error = (
                "Failed to start the camera. Another app may be using it."
            )
            return False

        self._cap = cap
        self._stop_event = threading.Event()
        capture_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_DEPTH)

        self._capture_worker = _CaptureWorker(cap, capture_queue, self._stop_event)
        self._processing_worker = _ProcessingWorker(
            self, capture_queue, self._stop_event, cap.actual_fps
        )
        self._capture_worker.start()
        self._processing_worker.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        for worker in (self._capture_worker, self._processing_worker):
            if worker is not None:
                worker.join(timeout=2.0)
        self._capture_worker = None
        self._processing_worker = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._fps = 0.0

    @property
    def is_running(self) -> bool:
        return (
            self._processing_worker is not None
            and self._processing_worker.is_alive()
            and not self._stop_event.is_set()
        )

    @property
    def has_failed(self) -> bool:
        """True when the camera dropped out while we thought we were live."""
        return self._cap is not None and self._stop_event.is_set()

    # ── observable state ─────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def error(self) -> str:
        return self._error

    @property
    def camera_resolution(self) -> tuple:
        if self._cap is None:
            return (0, 0)
        return (self._cap.actual_width, self._cap.actual_height)

    @property
    def camera_fps(self) -> float:
        return self._cap.actual_fps if self._cap is not None else 0.0

    def note_error(self, message: str) -> None:
        """Record a non-fatal problem for the UI, without spamming."""
        if message != self._error:
            self._error = message
            print(f"[live] {message}")

    def request_processor_reload(self) -> None:
        """Ask the loop to rebuild its processor chain on the next frame.

        Toggling an enhancer mid-stream changes which modules are active;
        this picks that up without restarting the camera.
        """
        self._reload_processors.set()

    def reload_processors_requested(self) -> bool:
        if self._reload_processors.is_set():
            self._reload_processors.clear()
            return True
        return False

    # ── sinks ────────────────────────────────────────────────────────────

    def add_sink(self, sink: FrameSink) -> None:
        with self._sink_lock:
            if sink not in self._sinks:
                self._sinks.append(sink)

    def remove_sink(self, sink: FrameSink) -> None:
        with self._sink_lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    def dispatch(self, frame: np.ndarray) -> None:
        """Hand a processed frame to every sink.

        A misbehaving sink must not take down the pipeline, so failures are
        swallowed per-sink; sinks surface their own errors through their own
        state instead.
        """
        with self._sink_lock:
            sinks = list(self._sinks)
        for sink in sinks:
            try:
                sink.send(frame)
            except Exception:
                pass
