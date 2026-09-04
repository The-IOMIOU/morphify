"""Helpers shared by the main window and the mapper dialogs.

Split out of ``ui.py`` so the dialogs and the redesigned main window can use
the same image conversion, translation, status routing and settings
persistence without importing each other.
"""

from __future__ import annotations

import json
import platform
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

import modules.globals
from modules.gettext import LanguageManager
from modules.paths import SETTINGS_PATH, ensure_user_dirs
from modules.gpu_processing import gpu_cvt_color, gpu_resize
from modules.utilities import has_image_extension

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph


# ─── module state ────────────────────────────────────────────────────────

_LANG: Optional[LanguageManager] = None
_APP: Optional[QApplication] = None
_STATUS_SINK: Optional[Callable[[str], None]] = None

# Last directory used per dialog kind, so file pickers reopen where the user
# left off. Shared between the main window and the mapper dialogs.
RECENT_DIRS: Dict[str, Optional[str]] = {
    "source": None,
    "target": None,
    "output": None,
}

SETTINGS_FILE = SETTINGS_PATH


def set_language(lang: str) -> LanguageManager:
    global _LANG
    _LANG = LanguageManager(lang)
    return _LANG


def _(text: str) -> str:
    """Translate via LanguageManager; falls back to identity."""
    if _LANG is None:
        return text
    return _LANG._(text)


# ─── status routing ──────────────────────────────────────────────────────


class _UIBridge(QObject):
    """Single QObject that owns cross-thread signals."""

    statusChanged = Signal(str)


_BRIDGE: Optional[_UIBridge] = None


def init_bridge(app: QApplication, sink: Callable[[str], None]) -> _UIBridge:
    """Wire status updates from any thread onto ``sink`` on the UI thread."""
    global _BRIDGE, _APP, _STATUS_SINK
    _APP = app
    _STATUS_SINK = sink
    _BRIDGE = _UIBridge()
    _BRIDGE.statusChanged.connect(sink)
    return _BRIDGE


def update_status(text: str) -> None:
    """Thread-safe status update — uses a signal if called off-UI thread."""
    message = _(text)
    if _BRIDGE is None:
        print(message)
        return
    _BRIDGE.statusChanged.emit(message)
    if _APP is not None and QThread.currentThread() is _APP.thread():
        # On UI thread — flush events so the user sees the update during
        # long synchronous start() runs.
        _APP.processEvents()


# ─── image utilities ─────────────────────────────────────────────────────


def fit_image_to_size(image, width: int, height: int):
    """BGR ndarray → BGR ndarray scaled to fit within (width, height)."""
    if width is None and height is None or width <= 0 or height <= 0:
        return image
    h, w = image.shape[:2]
    ratio = min(width / w, height / h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return gpu_resize(image, dsize=new_size)


def bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """BGR ndarray → QPixmap."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """PIL.Image → QPixmap."""
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimg = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def render_image_preview(image_path: str, size: Tuple[int, int]) -> QPixmap:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return pil_to_qpixmap(image)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> Optional[QPixmap]:
    capture = cv2.VideoCapture(video_path)
    try:
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if not has_frame:
            return None
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return pil_to_qpixmap(image)
    finally:
        capture.release()


def make_thumb(cv2_img: np.ndarray, size: int) -> QPixmap:
    rgb = gpu_cvt_color(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize((size, size), Image.LANCZOS)
    return pil_to_qpixmap(image)


# ─── persistence ─────────────────────────────────────────────────────────

# Keys mirrored between modules.globals and the settings file. Anything
# listed here is saved on change and restored at launch.
_PERSISTED_FLAGS = (
    "keep_fps",
    "keep_audio",
    "keep_frames",
    "many_faces",
    "map_faces",
    "poisson_blend",
    "color_correction",
    "nsfw_filter",
    "live_mirror",
    "live_resizable",
    "show_fps",
    "mouth_mask",
    "show_mouth_mask_box",
    "mouth_mask_size",
    "camera_width",
    "camera_height",
    "camera_fps",
    "detect_interval_ratio",
    "split_view",
    "performance_preset",
    "live_mode",
    "takeover_hair",
    "takeover_skin",
    "takeover_background",
    "takeover_skin_strength",
    "takeover_hair_volume",
    # bypass_swap is deliberately not persisted: it is a panic switch, and
    # starting a session silently un-swapped would be a nasty surprise.
    "virtual_cam_enabled",
    "virtual_cam_width",
    "virtual_cam_height",
    "virtual_cam_fps",
    "virtual_cam_mirror",
)


def save_switch_states() -> None:
    state: Dict[str, Any] = {key: getattr(modules.globals, key) for key in _PERSISTED_FLAGS}
    state["fp_ui"] = modules.globals.fp_ui
    ensure_user_dirs()
    try:
        with open(SETTINGS_FILE, "w") as handle:
            json.dump(state, handle, indent=2)
    except OSError:
        pass


def load_switch_states() -> None:
    try:
        with open(SETTINGS_FILE, "r") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return

    for key in _PERSISTED_FLAGS:
        if key in state:
            setattr(modules.globals, key, state[key])
    if isinstance(state.get("fp_ui"), dict):
        modules.globals.fp_ui.update(state["fp_ui"])


# ─── camera enumeration ──────────────────────────────────────────────────


def get_available_cameras() -> Tuple[List[int], List[str]]:
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if devices:
                return list(range(len(devices))), devices
            return [], ["No cameras found"]
        except Exception as exc:
            print(f"Error detecting cameras: {exc}")
            return [], ["No cameras found"]

    if platform.system() == "Darwin":
        return [0, 1], ["Camera 0", "Camera 1"]

    # Linux probe
    indices: List[int] = []
    names: List[str] = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            indices.append(i)
            names.append(f"Camera {i}")
            cap.release()
    return (indices, names) if names else ([], ["No cameras found"])


# ─── content check ───────────────────────────────────────────────────────


def check_and_ignore_nsfw(target, destroy: Optional[Callable] = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_frame, predict_image, predict_video

    check_nsfw = None
    if isinstance(target, str):
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("Processing ignored!")
        return True
    return False
