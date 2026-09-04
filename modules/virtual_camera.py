"""Publishes processed frames back out as a system camera device.

The rest of the app only ever produces frames; this module is the one place
that hands them to the OS so other applications (Discord, Zoom, Teams, OBS,
Chrome) can consume the swapped feed as an ordinary webcam.

On Windows this drives a DirectShow virtual-camera filter through
``pyvirtualcam``.  The filter has to be registered system-wide, which is why
OBS Studio (which ships and registers one) is a prerequisite — OBS itself
never has to be running, we only borrow its registered filter.

Two properties of the underlying device shape the design here:

* Resolution and frame rate are locked when the device is opened and cannot
  be renegotiated.  Frames that don't match are letterboxed to fit rather
  than triggering a reopen, which consumers see as a dropped device.
* ``send()`` is a blocking call into the filter.  It runs on a dedicated
  thread fed by a bounded queue so a stalled consumer can never back-pressure
  the face-swap loop.
"""

from __future__ import annotations

import platform
import queue
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

IS_WINDOWS = platform.system() == "Windows"

# DirectShow CLSID of the filter registered by the OBS Studio installer.
# Present in the registry whenever OBS >= 26.1 is installed, whether or not
# OBS is running.
_OBS_FILTER_CLSID = "{A3FCE0F5-3493-419F-958A-ABA1250EC20B}"

# Where DirectShow lists video capture devices.  Used to read back the
# friendly name the device will actually appear under in other apps.
_VIDEO_INPUT_CATEGORY = "{860BB310-5D01-11d0-BD3B-00A0C911CE86}"

OBS_DOWNLOAD_URL = "https://obsproject.com/download"

# Queue depth of 1: the outgoing feed should always show the newest frame.
# Anything deeper just adds latency between the preview and what viewers see.
_QUEUE_DEPTH = 1


def _probe_obs_filter() -> Tuple[bool, str]:
    """Check whether a usable virtual-camera filter is registered.

    Returns ``(available, reason)``.  ``reason`` is written for display in
    the UI, so it explains what to do rather than what failed.
    """
    if not IS_WINDOWS:
        # pyvirtualcam uses v4l2loopback on Linux and obs on macOS; we can't
        # probe those from the registry, so let the open attempt be the test.
        return True, "Backend availability is determined when starting."

    try:
        import winreg
    except ImportError:  # pragma: no cover - winreg always exists on Windows
        return False, "Could not access the Windows registry."

    try:
        with winreg.OpenKey(
            winreg.HKEY_CLASSES_ROOT,
            rf"CLSID\{_OBS_FILTER_CLSID}\InprocServer32",
        ) as key:
            dll_path, _ = winreg.QueryValueEx(key, "")
    except OSError:
        return False, (
            "OBS Virtual Camera is not installed. Install OBS Studio once "
            "(it does not need to be running) to enable camera output."
        )

    if not dll_path:
        return False, (
            "OBS Virtual Camera is registered but its driver path is empty. "
            "Reinstalling OBS Studio will repair it."
        )

    return True, "OBS Virtual Camera is installed and ready."


def _read_device_name() -> str:
    """Read the name the virtual camera appears under in other apps."""
    default = "OBS Virtual Camera"
    if not IS_WINDOWS:
        return default
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CLASSES_ROOT,
            rf"CLSID\{_VIDEO_INPUT_CATEGORY}\Instance\{_OBS_FILTER_CLSID}",
        ) as key:
            name, _ = winreg.QueryValueEx(key, "FriendlyName")
            return name or default
    except OSError:
        return default


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale ``frame`` into a ``width`` x ``height`` canvas, preserving aspect.

    The device resolution is fixed at open time while the camera (or the
    user's chosen capture size) may differ, so frames are fitted with black
    bars instead of being stretched.
    """
    h, w = frame.shape[:2]
    if w == width and h == height:
        return frame

    scale = min(width / w, height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    if new_w == width and new_h == height:
        return resized

    canvas = np.zeros((height, width, 3), dtype=frame.dtype)
    top = (height - new_h) // 2
    left = (width - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


class VirtualCamSink:
    """A frame sink that publishes to a system virtual camera device.

    Safe to call ``send()`` on from the processing thread at any time: it is
    a no-op while stopped and never blocks.
    """

    def __init__(self) -> None:
        self._cam = None
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_DEPTH)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._width = 0
        self._height = 0
        self._fps = 30.0
        self._device_name = ""
        self._error = ""
        self._frames_sent = 0
        # Set by the UI when the outgoing feed should be flipped relative to
        # the frames it receives (the preview and the published feed often
        # want opposite handedness).
        self.mirror = False

    # ── state ────────────────────────────────────────────────────────────

    @staticmethod
    def available() -> Tuple[bool, str]:
        """Whether a virtual-camera backend is installed, and a reason why."""
        return _probe_obs_filter()

    @property
    def is_running(self) -> bool:
        return self._cam is not None

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def error(self) -> str:
        return self._error

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._width, self._height

    @property
    def frames_sent(self) -> int:
        return self._frames_sent

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self, width: int, height: int, fps: float = 30.0) -> bool:
        """Open the device. Returns False and sets ``error`` on failure."""
        with self._lock:
            if self._cam is not None:
                return True

            self._error = ""
            ok, reason = self.available()
            if not ok:
                self._error = reason
                return False

            try:
                import pyvirtualcam
            except ImportError:
                self._error = (
                    "pyvirtualcam is not installed. Run: "
                    "pip install pyvirtualcam"
                )
                return False

            # Even dimensions: the OBS filter delivers NV12, whose chroma
            # planes are half-resolution, so odd sizes are rejected.
            width = max(2, width - (width % 2))
            height = max(2, height - (height % 2))
            fps = float(max(1.0, min(fps, 120.0)))

            try:
                self._cam = pyvirtualcam.Camera(
                    width=width,
                    height=height,
                    fps=fps,
                    fmt=pyvirtualcam.PixelFormat.BGR,
                    backend="obs" if IS_WINDOWS else None,
                    print_fps=False,
                )
            except Exception as exc:
                self._cam = None
                self._error = self._explain_open_failure(exc)
                return False

            self._width = width
            self._height = height
            self._fps = fps
            self._frames_sent = 0
            self._device_name = getattr(self._cam, "device", "") or _read_device_name()

            # Drain anything left from a previous run.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

            self._stop.clear()
            self._thread = threading.Thread(
                target=self._pump, name="VirtualCamSink", daemon=True
            )
            self._thread.start()
            return True

    def stop(self) -> None:
        """Close the device and release it back to the system."""
        with self._lock:
            if self._cam is None and self._thread is None:
                return
            self._stop.set()
            thread, self._thread = self._thread, None

        if thread is not None:
            thread.join(timeout=2.0)

        with self._lock:
            cam, self._cam = self._cam, None
            if cam is not None:
                try:
                    cam.close()
                except Exception:
                    pass
            self._device_name = ""
            self._width = self._height = 0

    # ── frame path ───────────────────────────────────────────────────────

    def send(self, bgr_frame: np.ndarray) -> None:
        """Queue a BGR frame for the device. Never blocks; drops if behind."""
        if self._cam is None or bgr_frame is None:
            return
        try:
            self._queue.put_nowait(bgr_frame)
        except queue.Full:
            # Newest frame wins — replace whatever is waiting.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(bgr_frame)
            except queue.Full:
                pass

    def _pump(self) -> None:
        """Feed queued frames to the device on a dedicated thread."""
        # Cache the send-ready buffer, not the source frame: re-sending a
        # frame must not mirror or rescale it a second time.
        last_fitted: Optional[np.ndarray] = None
        while not self._stop.is_set():
            fitted = None
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                # Nothing new. Re-send the last frame so consumers see a live
                # feed rather than a frozen or timed-out device.
                fitted = last_fitted
                if fitted is None:
                    continue

            cam = self._cam
            if cam is None:
                break

            try:
                if fitted is None:
                    if self.mirror:
                        frame = cv2.flip(frame, 1)
                    fitted = letterbox(frame, self._width, self._height)
                    if not fitted.flags["C_CONTIGUOUS"]:
                        fitted = np.ascontiguousarray(fitted)
                    last_fitted = fitted
                cam.send(fitted)
                cam.sleep_until_next_frame()
                self._frames_sent += 1
            except Exception as exc:
                self._error = f"Virtual camera stopped: {exc}"
                break

    # ── diagnostics ──────────────────────────────────────────────────────

    @staticmethod
    def _explain_open_failure(exc: Exception) -> str:
        """Turn a pyvirtualcam exception into something actionable."""
        text = str(exc).lower()
        if "not installed" in text or "no such file" in text or "backend" in text:
            return (
                "No virtual camera driver found. Install OBS Studio once to "
                "register one; it does not need to be running."
            )
        if "in use" in text or "busy" in text or "already" in text:
            return (
                "The virtual camera is already in use. Close OBS or any other "
                "app holding it, then try again."
            )
        return f"Could not start the virtual camera: {exc}"
