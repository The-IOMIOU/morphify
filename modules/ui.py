"""Main application window.

Layout is a fixed header (wordmark + live status pills), a left nav rail, and
a stacked page area:

    Live    - camera picker, embedded preview, virtual-cam toggle, quick controls
    Faces   - the source-face library
    Studio  - batch face swap onto an image or video file
    Setup   - capture, virtual camera, processing and output settings
    About   - version, models, licence

The window drives ``LiveEngine`` (capture + swap threads) and registers two
sinks on it: one for the on-screen preview and one for the virtual camera.
Neither can stall the other, and adding a third consumer is a one-liner.

``core.py`` depends on three names from this module — ``init``,
``update_status`` and ``check_and_ignore_nsfw`` — which keep their original
signatures.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import webbrowser
from typing import Callable, List, Optional

import cv2
import requests
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

import modules.globals
import modules.metadata
from modules import imwrite_unicode, recorder, ui_dialogs
from modules.ui_nav import FACES_PAGE_INDEX, MOTION_PAGE_INDEX, NAV_ITEMS
from modules.face_analyser import (
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
)
from modules.live_engine import LatestFrameSink, LiveEngine
from modules import paths
from modules.paths import CAPTURES_DIR, FACES_DIR, MODELS_DIR, ensure_user_dirs
from modules.processors.frame.core import get_frame_processors_modules
from modules.ui_common import (
    RECENT_DIRS,
    _,
    bgr_to_qpixmap,
    check_and_ignore_nsfw,  # noqa: F401 - re-exported for core.py
    fit_image_to_size,
    get_available_cameras,
    init_bridge,
    load_switch_states,
    render_image_preview,
    render_video_preview,
    save_switch_states,
    set_language,
    update_status,  # noqa: F401 - re-exported for core.py
)
from modules.ui_theme import (
    FACE_THUMB_SIZE,
    SIDEBAR_WIDTH,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    stylesheet,
)
from modules.utilities import is_image, is_video
from modules.virtual_camera import OBS_DOWNLOAD_URL, VirtualCamSink

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.gif *.bmp *.webp)"
MEDIA_FILTER = "Media (*.png *.jpg *.jpeg *.gif *.bmp *.webp *.mp4 *.mkv *.mov *.avi)"

DROP_PREVIEW_SIZE = 168

# The site's root URL now returns an HTML page; the image lives here.
RANDOM_FACE_URL = "https://thispersondoesnotexist.com/random-person.jpeg"

CAPTURE_PRESETS = [
    ("640 x 360", 640, 360),
    ("854 x 480", 854, 480),
    ("960 x 540", 960, 540),
    ("1280 x 720", 1280, 720),
    ("1920 x 1080", 1920, 1080),
]

VCAM_PRESETS = [
    ("640 x 480", 640, 480),
    ("854 x 480", 854, 480),
    ("1280 x 720", 1280, 720),
    ("1920 x 1080", 1920, 1080),
]

# One dial for the frame-rate/fidelity trade. detect_ratio is the fraction of
# frames that run full detection; the rest reuse the last result.
PERFORMANCE_PRESETS = {
    "Performance": {
        "camera": (640, 360),
        "detect_ratio": 0.14,
        "enhancer": "None",
        "hint": "Highest frame rate. Detection is less frequent, so fast head "
                "movement lags slightly.",
    },
    "Balanced": {
        "camera": (960, 540),
        "detect_ratio": 0.08,
        "enhancer": "None",
        "hint": "The default. Smooth on a mid-range GPU.",
    },
    "Quality": {
        "camera": (1280, 720),
        "detect_ratio": 0.05,
        "enhancer": "GPEN-256",
        "hint": "Sharper and tracks better, at a real cost in frame rate.",
    },
    "Custom": {
        "camera": None,
        "detect_ratio": None,
        "enhancer": None,
        "hint": "Your own settings.",
    },
}

# The ways to wear someone else's appearance. Kept as a table so the strip,
# the tooltips and the saved setting cannot drift apart.
# Two modes were built and withdrawn after testing on a real webcam:
#
#   Full Takeover      transplanted hair read as a pasted cutout with seams.
#   Portrait animation LivePortrait; the generated head composited into the
#                      live frame looked like a cheap face filter.
#
# Both remain in the tree (modules/takeover.py, modules/live_portrait.py)
# but neither is offered, because a mode that disappoints is worse than a
# mode that is absent.
MODES = [
    ("swap", _("Face swap"),
     "Replaces the face only. Fastest and the most robust to head movement — "
     "you keep your own hair, skin tone and room."),
]

ENHANCER_CHOICES = ["None", "GFPGAN", "GPEN-512", "GPEN-256"]
ENHANCER_KEYS = {
    "None": None,
    "GFPGAN": "face_enhancer",
    "GPEN-512": "face_enhancer_gpen512",
    "GPEN-256": "face_enhancer_gpen256",
}

_APP: Optional[QApplication] = None
_MAIN: Optional["MainWindow"] = None


# ─── small widgets ───────────────────────────────────────────────────────


class Pill(QLabel):
    """A compact status chip in the header.

    ``state`` drives the colour through the stylesheet's property selectors
    rather than an inline stylesheet, so all the colours stay in one place.
    """

    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setProperty("pill", True)
        self.setProperty("state", "idle")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set(self, text: str, state: str = "idle") -> None:
        if self.text() == text and self.property("state") == state:
            return
        self.setText(text)
        self.setProperty("state", state)
        # Property selectors are only re-evaluated on an explicit repolish.
        self.style().unpolish(self)
        self.style().polish(self)


def card(title: str = "") -> tuple:
    """A titled panel. Returns (frame, content_layout)."""
    frame = QFrame()
    frame.setObjectName("card")
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(16, 14, 16, 16)
    outer.setSpacing(12)
    if title:
        label = QLabel(title.upper())
        label.setObjectName("cardTitle")
        outer.addWidget(label)
    return frame, outer


def separator() -> QFrame:
    line = QFrame()
    line.setObjectName("separator")
    line.setFrameShape(QFrame.Shape.HLine)
    return line


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def compose_split(original, processed):
    """Original on the left, swapped on the right, with a divider.

    Preview-only: the virtual camera and the recorder still receive the
    plain processed frame, because nobody wants a split screen published to
    their call.
    """
    if original is None or processed is None:
        return processed
    if original.shape != processed.shape:
        return processed

    height, width = processed.shape[:2]
    half = width // 2
    combined = processed.copy()
    combined[:, :half] = original[:, :half]
    # A hairline so the seam reads as intentional rather than as a glitch.
    cv2.line(combined, (half, 0), (half, height), (255, 255, 255), 1)
    return combined


def app_icon_path() -> str:
    """Locate the bundled application icon, or "" if it is not there.

    PyInstaller drops data files beside the executable in ``_internal``;
    a source checkout keeps it under ``packaging``.
    """
    candidates = []
    if getattr(sys, "_MEIPASS", None):
        candidates.append(os.path.join(sys._MEIPASS, "Morphify.ico"))
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(here, "packaging", "Morphify.ico"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def _open_folder(path: str) -> None:
    """Reveal a folder in the system file manager."""
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - opening a local folder
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class DropLabel(QLabel):
    """Image well that accepts a dropped or pasted image file."""

    def __init__(self, placeholder: str, size: int = DROP_PREVIEW_SIZE):
        super().__init__(placeholder)
        self._placeholder = placeholder
        self.setObjectName("imageDrop")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setFixedSize(size, size)
        self.setAcceptDrops(True)
        self.on_file: Optional[Callable[[str], None]] = None

    def clear_image(self) -> None:
        self.clear()
        self.setText(self._placeholder)

    def _set_drag_active(self, active: bool) -> None:
        self.setProperty("dragActive", active)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_drag_active(True)

    def dragLeaveEvent(self, event) -> None:
        self._set_drag_active(False)

    def dropEvent(self, event) -> None:
        self._set_drag_active(False)
        urls = event.mimeData().urls()
        if not urls or self.on_file is None:
            return
        path = urls[0].toLocalFile()
        if path:
            self.on_file(path)
            event.acceptProposedAction()


class LabelledSlider(QWidget):
    """Slider with a caption and a live value readout."""

    def __init__(self, caption: str, minimum: float, maximum: float,
                 value: float, denominator: int, fmt: str = "{:.2f}",
                 tooltip: str = ""):
        super().__init__()
        self._denominator = denominator
        self._fmt = fmt
        self.on_change: Optional[Callable[[float], None]] = None

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(2)

        self._caption = QLabel(caption)
        self._caption.setObjectName("hint")
        self._readout = QLabel(fmt.format(value))
        self._readout.setObjectName("hint")
        self._readout.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(minimum * denominator), int(maximum * denominator))
        self.slider.setValue(int(value * denominator))
        self.slider.valueChanged.connect(self._emit)
        if tooltip:
            self.slider.setToolTip(tooltip)
            self._caption.setToolTip(tooltip)

        layout.addWidget(self._caption, 0, 0)
        layout.addWidget(self._readout, 0, 1)
        layout.addWidget(self.slider, 1, 0, 1, 2)

    def _emit(self, raw: int) -> None:
        value = raw / self._denominator
        self._readout.setText(self._fmt.format(value))
        if self.on_change:
            self.on_change(value)

    def value(self) -> float:
        return self.slider.value() / self._denominator


def toggle(label: str, field: str, tooltip: str = "",
           extra: Optional[Callable[[bool], None]] = None) -> QCheckBox:
    """Checkbox bound to a ``modules.globals`` flag, persisted on change."""
    box = QCheckBox(label)
    box.setChecked(bool(getattr(modules.globals, field)))
    if tooltip:
        box.setToolTip(tooltip)

    def handler(checked: bool) -> None:
        setattr(modules.globals, field, checked)
        save_switch_states()
        if extra:
            extra(checked)

    box.toggled.connect(handler)
    return box


# ─── main window ─────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    def __init__(self, start_cb: Callable, destroy_cb: Callable):
        super().__init__()
        load_switch_states()
        self._start_cb = start_cb
        self._destroy_cb = destroy_cb

        self._engine = LiveEngine()
        self._preview_sink = LatestFrameSink()
        self._vcam = VirtualCamSink()
        self._recorder = recorder.RecorderSink()
        self._preview: Optional[ui_dialogs.PreviewWindow] = None
        self._motion_dialog = None
        self._face_buttons: List[QPushButton] = []
        self._face_paths: List[str] = []
        self._face_filter = ""
        self._shutting_down = False
        self._tray: Optional[QSystemTrayIcon] = None
        self._live_started_at = 0.0
        # Set by the tray's Quit action so closeEvent knows the
        # difference between 'hide me' and 'actually exit'.
        self._force_quit = False

        self.setWindowTitle(
            f"{modules.metadata.name} {modules.metadata.version}"
        )
        icon_path = app_icon_path()
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())

        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_live_page())
        self._pages.addWidget(self._build_faces_page())
        self._pages.addWidget(self._build_studio_page())
        self._pages.addWidget(self._build_motion_page())
        self._pages.addWidget(self._build_setup_page())
        self._pages.addWidget(self._build_about_page())
        body.addWidget(self._pages, 1)
        outer.addLayout(body, 1)

        # Repaint the preview and refresh the header on a timer rather than
        # per frame: the pipeline runs faster than the display needs to.
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._tick)
        self._ui_timer.start(16)

        self._build_shortcuts()
        self._build_tray()
        self._refresh_vcam_availability()
        self._reload_face_library()
        self._sync_source_previews()

    # ── shortcuts ────────────────────────────────────────────────────────

    # Keeping every binding in one table makes the About page's cheat sheet
    # impossible to drift out of sync with what is actually wired up.
    def _shortcut_table(self) -> list:
        return [
            ("Ctrl+L", _("Go live / stop"), lambda: self.btn_live.click()),
            ("Ctrl+K", _("Virtual camera on / off"), self._toggle_vcam_shortcut),
            ("Ctrl+R", _("Start / stop recording"), lambda: self.btn_record.click()),
            ("Ctrl+S", _("Save a snapshot"), self._on_snapshot),
            ("Ctrl+B", _("Bypass the swap (panic)"),
             lambda: self.sw_bypass.setChecked(not self.sw_bypass.isChecked())),
            ("Ctrl+D", _("Split view on / off"),
             lambda: self.sw_split.setChecked(not self.sw_split.isChecked())),
            ("Ctrl+M", _("Mirror the preview"), self._toggle_mirror_shortcut),
            ("]", _("Next face"), lambda: self._step_face(1)),
            ("[", _("Previous face"), lambda: self._step_face(-1)),
            ("Ctrl+1", _("Live page"), lambda: self._go_to_page(0)),
            ("Ctrl+2", _("Faces page"), lambda: self._go_to_page(1)),
            ("Ctrl+3", _("Studio page"), lambda: self._go_to_page(2)),
            ("Ctrl+4", _("Setup page"), lambda: self._go_to_page(3)),
            ("Ctrl+5", _("About page"), lambda: self._go_to_page(4)),
        ]

    def _build_shortcuts(self) -> None:
        for sequence, _label, slot in self._shortcut_table():
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(slot)

    def _go_to_page(self, index: int) -> None:
        self._pages.setCurrentIndex(index)
        self._nav_group.button(index).setChecked(True)

    def _toggle_vcam_shortcut(self) -> None:
        if self.btn_vcam.isEnabled():
            self.btn_vcam.click()

    def _toggle_mirror_shortcut(self) -> None:
        modules.globals.live_mirror = not modules.globals.live_mirror
        save_switch_states()
        update_status(
            f"Preview mirror {'on' if modules.globals.live_mirror else 'off'}")

    def _step_face(self, delta: int) -> None:
        """Move to the next/previous face in the library."""
        if not self._face_paths:
            update_status("No faces in the library yet")
            return
        current = modules.globals.source_path
        try:
            index = self._face_paths.index(os.path.abspath(current or ""))
        except ValueError:
            index = -1 if delta > 0 else 0
        index = (index + delta) % len(self._face_paths)
        self._apply_source_path(self._face_paths[index])

    # ── system tray ──────────────────────────────────────────────────────

    def _build_tray(self) -> None:
        """Keep the app reachable while its window is out of the way.

        Live mode is usually running behind a call, so closing to the tray
        rather than quitting is the behaviour that matches how it is used.
        """
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        icon = self.windowIcon()
        if icon.isNull():
            icon = QIcon(app_icon_path()) if app_icon_path() else QIcon()
        self._tray = QSystemTrayIcon(icon, self)
        self._tray.setToolTip(modules.metadata.name)

        menu = QMenu()
        self._act_show = menu.addAction(_("Show window"))
        self._act_show.triggered.connect(self._restore_window)
        self._act_live = menu.addAction(_("Go live"))
        self._act_live.triggered.connect(lambda: self.btn_live.click())
        self._act_vcam = menu.addAction(_("Virtual camera"))
        self._act_vcam.triggered.connect(self._toggle_vcam_shortcut)
        menu.addSeparator()
        act_quit = menu.addAction(_("Quit"))
        act_quit.triggered.connect(self._quit_from_tray)

        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._restore_window()

    def _restore_window(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _quit_from_tray(self) -> None:
        self._force_quit = True
        self.close()

    # ── header ───────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(52)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 0, 18, 0)
        layout.setSpacing(10)

        mark = QLabel("MORPH")
        mark.setObjectName("wordmark")
        mark_accent = QLabel("IFY")
        mark_accent.setObjectName("wordmarkAccent")
        layout.addWidget(mark)
        layout.addWidget(mark_accent)
        layout.addStretch(1)

        self.pill_status = QLabel("")
        self.pill_status.setObjectName("statusLabel")
        layout.addWidget(self.pill_status)
        layout.addSpacing(8)

        self.pill_device = Pill("")
        self.pill_vcam = Pill("VCAM OFF")
        self.pill_fps = Pill("— FPS")
        for pill in (self.pill_device, self.pill_vcam, self.pill_fps):
            layout.addWidget(pill)

        return header

    # ── sidebar ──────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("sidebar")
        bar.setFixedWidth(SIDEBAR_WIDTH)
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(0, 12, 0, 12)
        layout.setSpacing(2)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        for index, (icon, title) in enumerate(NAV_ITEMS):
            button = QPushButton(f"  {icon}   {_(title)}")
            button.setObjectName("navItem")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _c, i=index: self._pages.setCurrentIndex(i))
            self._nav_group.addButton(button, index)
            layout.addWidget(button)
        self._nav_group.button(0).setChecked(True)

        layout.addStretch(1)

        # Current source face, always visible regardless of page.
        chip = QVBoxLayout()
        chip.setContentsMargins(16, 0, 16, 6)
        chip.setSpacing(6)
        caption = QLabel(_("SOURCE FACE"))
        caption.setObjectName("cardTitle")
        chip.addWidget(caption)
        self.sidebar_face = DropLabel(_("No face"), SIDEBAR_WIDTH - 32)
        self.sidebar_face.on_file = self._apply_source_path
        chip.addWidget(self.sidebar_face)
        layout.addLayout(chip)

        return bar

    # ── live page ────────────────────────────────────────────────────────

    def _build_live_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        # Mode strip: the app has several ways to wear someone else's face,
        # and they need to be visible rather than buried in a settings page.
        layout.addWidget(self._build_mode_strip())

        self.video_surface = QLabel(
            _("Select a source face and a camera, then press Go Live.")
        )
        self.video_surface.setObjectName("videoSurface")
        self.video_surface.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_surface.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.video_surface.setMinimumHeight(300)
        layout.addWidget(self.video_surface, 1)

        # Transport row.
        controls = QHBoxLayout()
        controls.setSpacing(10)

        self.cb_camera = QComboBox()
        self.cb_camera.setMinimumWidth(220)
        self._camera_indices: List[int] = []
        self._reload_cameras()
        controls.addWidget(self.cb_camera)

        btn_refresh = QPushButton("⟳")
        btn_refresh.setObjectName("iconButton")
        btn_refresh.setFixedSize(38, 38)
        btn_refresh.setToolTip(_("Rescan for cameras"))
        btn_refresh.clicked.connect(self._reload_cameras)
        controls.addWidget(btn_refresh)

        self.btn_live = QPushButton(_("▶  Go Live"))
        self.btn_live.setObjectName("primary")
        self.btn_live.setCheckable(True)
        self.btn_live.setMinimumWidth(140)
        self.btn_live.setToolTip(_("Start the real-time face swap on this camera"))
        self.btn_live.clicked.connect(self._on_live_toggled)
        controls.addWidget(self.btn_live)

        self.btn_vcam = QPushButton(_("◉  Virtual Camera"))
        self.btn_vcam.setObjectName("toggle")
        self.btn_vcam.setCheckable(True)
        self.btn_vcam.setMinimumWidth(170)
        self.btn_vcam.clicked.connect(self._on_vcam_toggled)
        controls.addWidget(self.btn_vcam)

        self.btn_record = QPushButton(_("●  Record"))
        self.btn_record.setObjectName("toggle")
        self.btn_record.setCheckable(True)
        self.btn_record.setMinimumWidth(120)
        self.btn_record.setToolTip(_("Record the swapped feed to a video file  (Ctrl+R)"))
        self.btn_record.clicked.connect(self._on_record_toggled)
        controls.addWidget(self.btn_record)

        self.btn_snapshot = QPushButton(_("⛶  Snap"))
        self.btn_snapshot.setToolTip(_("Save the current frame as a picture  (Ctrl+S)"))
        self.btn_snapshot.clicked.connect(self._on_snapshot)
        controls.addWidget(self.btn_snapshot)

        controls.addStretch(1)
        layout.addLayout(controls)

        # Second row: view and safety toggles.
        toggles = QHBoxLayout()
        toggles.setSpacing(16)
        self.sw_bypass = toggle(
            _("Bypass swap"), "bypass_swap",
            _("Send your real face through instantly, without stopping the "
              "stream. Panic button.  (Ctrl+B)"))
        toggles.addWidget(self.sw_bypass)
        self.sw_split = toggle(
            _("Split view"), "split_view",
            _("Show the original next to the swap, in the preview only"))
        toggles.addWidget(self.sw_split)
        toggles.addWidget(toggle(
            _("Mirror preview"), "live_mirror",
            _("Flip the preview horizontally, like a mirror")))
        toggles.addWidget(toggle(
            _("Show FPS"), "show_fps",
            _("Draw the frame rate onto the video")))
        toggles.addStretch(1)
        self.lbl_session = QLabel("")
        self.lbl_session.setObjectName("hint")
        toggles.addWidget(self.lbl_session)
        layout.addLayout(toggles)

        self.lbl_vcam_hint = QLabel("")
        self.lbl_vcam_hint.setObjectName("hint")
        self.lbl_vcam_hint.setWordWrap(True)
        layout.addWidget(self.lbl_vcam_hint)

        # Quick controls.
        quick_frame, quick = card(_("Quick controls"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(10)

        grid.addWidget(QLabel(_("Preset")), 0, 0)
        self.cb_preset = QComboBox()
        self.cb_preset.addItems(list(PERFORMANCE_PRESETS))
        self.cb_preset.setCurrentText(modules.globals.performance_preset)
        self.cb_preset.currentTextChanged.connect(self._on_preset_change)
        self.cb_preset.setToolTip(
            _("Trades frame rate against fidelity in one move. Sets the "
              "capture size, how often faces are re-detected, and the "
              "enhancer."))
        grid.addWidget(self.cb_preset, 0, 1)

        grid.addWidget(QLabel(_("Face enhancer")), 1, 0)
        self.cb_enhancer = QComboBox()
        self.cb_enhancer.addItems(ENHANCER_CHOICES)
        self.cb_enhancer.setCurrentText(self._current_enhancer())
        self.cb_enhancer.currentTextChanged.connect(self._on_enhancer_change)
        self.cb_enhancer.setToolTip(
            _("Sharpens the swapped face. Costs frame rate — GPEN-256 is the "
              "cheapest, GFPGAN the most expensive.")
        )
        grid.addWidget(self.cb_enhancer, 1, 1)

        self.sw_many_faces = toggle(
            _("Swap every face"), "many_faces",
            _("Swap all detected faces instead of only the largest"))
        grid.addWidget(self.sw_many_faces, 0, 2)

        self.sw_blend = toggle(
            _("Blend identity"), "blend_identity",
            _("Average the identity across every photo sharing a name "
              "(kai-cenat-01, -02, ...). Steadier likeness than one photo. "
              "Photos of a different person are detected and left out."),
            extra=lambda _v: self._on_blend_changed())
        grid.addWidget(self.sw_blend, 3, 2)

        self.sl_opacity = LabelledSlider(
            _("Opacity"), 0.0, 1.0, modules.globals.opacity, 100, "{:.0%}",
            _("Blend between the original and the swapped face"))
        self.sl_opacity.on_change = self._on_opacity_change
        grid.addWidget(self.sl_opacity, 2, 0, 1, 2)

        self.sl_sharpness = LabelledSlider(
            _("Sharpness"), 0.0, 5.0, modules.globals.sharpness, 10, "{:.1f}",
            _("Sharpen the swapped face"))
        self.sl_sharpness.on_change = self._on_sharpness_change
        grid.addWidget(self.sl_sharpness, 1, 2)

        self.sl_likeness = LabelledSlider(
            _("Likeness"), 0.0, 1.0, modules.globals.identity_strength, 100,
            "{:.0%}",
            _("Pushes the swap away from your own identity. Measured: it "
              "removes your features steadily, but past roughly 40% it also "
              "drifts away from the source, so it stops adding likeness and "
              "starts distorting. 20-40% is the useful range.\n\n"
              "It cannot change face shape, jaw or hairline — those stay "
              "yours no matter what, because the swapper only replaces the "
              "inner face."))
        self.sl_likeness.on_change = self._on_likeness_change
        grid.addWidget(self.sl_likeness, 2, 2)

        self.sl_mouth = LabelledSlider(
            _("Mouth mask"), 0.0, 100.0, 0.0, 1, "{:.0f}",
            _("0 keeps the swapped mouth; higher values expose your real "
              "mouth down toward the chin"))
        self.sl_mouth.on_change = self._on_mouth_mask_change
        self.sl_mouth.slider.sliderPressed.connect(self._on_mouth_mask_pressed)
        self.sl_mouth.slider.sliderReleased.connect(self._on_mouth_mask_released)
        grid.addWidget(self.sl_mouth, 3, 0, 1, 2)

        quick.addLayout(grid)
        layout.addWidget(quick_frame)
        return page

    # ── mode strip ───────────────────────────────────────────────────────

    def _build_mode_strip(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("card")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(14, 10, 14, 12)
        outer.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(8)
        caption = QLabel(_("MODE"))
        caption.setObjectName("cardTitle")
        row.addWidget(caption)
        row.addSpacing(10)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for index, (key, title, blurb) in enumerate(MODES):
            button = QPushButton(title)
            button.setObjectName("modeChip")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(blurb)
            button.setProperty("modeKey", key)
            self._mode_group.addButton(button, index)
            row.addWidget(button)
        row.addStretch(1)
        outer.addLayout(row)

        self.lbl_mode = QLabel("")
        self.lbl_mode.setObjectName("hint")
        self.lbl_mode.setWordWrap(True)
        outer.addWidget(self.lbl_mode)

        # Options that only make sense for Full Takeover.
        self.takeover_options = QWidget()
        options = QHBoxLayout(self.takeover_options)
        options.setContentsMargins(0, 2, 0, 0)
        options.setSpacing(16)
        options.addWidget(toggle(
            _("Their hair"), "takeover_hair",
            _("Segment their hair and warp it onto your head. Best at "
              "roughly frontal angles.")))
        options.addWidget(toggle(
            _("Their skin tone"), "takeover_skin",
            _("Match your neck and shoulders to the new face.")))
        options.addWidget(toggle(
            _("Their background"), "takeover_background",
            _("Cut you out and place you in the photo's own setting.")))

        self.sl_skin = LabelledSlider(
            _("Tone strength"), 0.0, 1.0,
            modules.globals.takeover_skin_strength, 100, "{:.0%}",
            _("How far to push your skin toward theirs"))
        self.sl_skin.on_change = self._on_skin_strength
        self.sl_skin.setMaximumWidth(200)
        options.addWidget(self.sl_skin)

        self.sl_volume = LabelledSlider(
            _("Hair swing"), 0.0, 2.0,
            modules.globals.takeover_hair_volume, 50, "{:.1f}",
            _("Leans the hair with your head turn to fake depth. It is a 2D "
              "cutout, so this is an illusion, not real volume."))
        self.sl_volume.on_change = self._on_hair_volume
        self.sl_volume.setMaximumWidth(200)
        options.addWidget(self.sl_volume)
        options.addStretch(1)
        outer.addWidget(self.takeover_options)

        # Portrait animation controls.
        self.portrait_options = QWidget()
        portrait = QHBoxLayout(self.portrait_options)
        portrait.setContentsMargins(0, 2, 0, 0)
        portrait.setSpacing(12)
        btn_neutral = QPushButton(_("Set neutral pose"))
        btn_neutral.setToolTip(
            _("Take your current pose as the resting one. Do this facing the "
              "camera straight on — everything after is measured against it."))
        btn_neutral.clicked.connect(self._on_reset_reference)
        portrait.addWidget(btn_neutral)
        note = QLabel(_(
            "Only the head is generated. Expect around 10 fps — this model "
            "renders a face rather than pasting one."))
        note.setObjectName("hint")
        note.setWordWrap(True)
        portrait.addWidget(note, 1)
        outer.addWidget(self.portrait_options)

        self._mode_group.idClicked.connect(self._on_mode_changed)
        self._mode_group.button(self._current_mode_index()).setChecked(True)
        self._apply_mode(MODES[self._current_mode_index()][0], announce=False)
        return frame

    @staticmethod
    def _current_mode_index() -> int:
        for index, (key, _title, _blurb) in enumerate(MODES):
            if key == modules.globals.live_mode:
                return index
        return 0

    def _on_mode_changed(self, index: int) -> None:
        self._apply_mode(MODES[index][0])

    def _apply_mode(self, key: str, announce: bool = True) -> None:
        modules.globals.live_mode = key
        modules.globals.takeover_enabled = (key == "takeover")
        modules.globals.portrait_enabled = (key == "portrait")
        self.takeover_options.setVisible(key == "takeover")
        self.portrait_options.setVisible(key == "portrait")

        # Keep the chip in step when the mode is set in code (restored from
        # settings, or driven by a shortcut) rather than by a click.
        for index, (candidate, _title, _blurb) in enumerate(MODES):
            if candidate == key:
                button = self._mode_group.button(index)
                if button is not None and not button.isChecked():
                    button.setChecked(True)
                break

        blurb = next(b for k, _t, b in MODES if k == key)
        if key == "takeover":
            from modules import takeover
            if not takeover.models_available():
                blurb += "  " + _("Models for this mode are missing — "
                                  "Setup > Models > Download missing.")
        elif key == "portrait":
            from modules import live_portrait
            if not live_portrait.models_available():
                blurb += "  " + _("Models for this mode are missing — "
                                  "Setup > Models > Download missing.")
        self.lbl_mode.setText(blurb)

        save_switch_states()
        self._engine.request_processor_reload()
        if announce:
            update_status(next(t for k, t, _b in MODES if k == key))

    def _on_reset_reference(self) -> None:
        from modules import live_portrait
        live_portrait.cache.invalidate()
        self._engine.request_processor_reload()
        update_status("Neutral pose will be taken from the next frame")

    def _on_skin_strength(self, value: float) -> None:
        modules.globals.takeover_skin_strength = value

    def _on_hair_volume(self, value: float) -> None:
        modules.globals.takeover_hair_volume = value

    # ── faces page ───────────────────────────────────────────────────────

    def _build_faces_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel(_("Faces"))
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        hint = QLabel(
            _("Click a face to swap to it — you can switch while you are live "
              "( [ and ] step through them ). Right-click a face for more. "
              "Starred faces sort first.")
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        btn_find = QPushButton(_("🔍  Find faces"))
        btn_find.setObjectName("primary")
        btn_find.setToolTip(
            _("Generate synthetic portraits, or fetch one from a link"))
        btn_find.clicked.connect(self._on_find_faces)
        actions.addWidget(btn_find)

        btn_import = QPushButton(_("＋  Import"))
        btn_import.setToolTip(_("Copy face images in from disk"))
        btn_import.clicked.connect(self._on_import_faces)
        actions.addWidget(btn_import)

        btn_random = QPushButton(_("🎲  Random"))
        btn_random.setToolTip(
            _("Add one synthetic face from thispersondoesnotexist.com"))
        btn_random.clicked.connect(self._on_random_face)
        actions.addWidget(btn_random)

        btn_folder = QPushButton(_("Open folder"))
        btn_folder.clicked.connect(self._on_open_faces_folder)
        actions.addWidget(btn_folder)

        btn_rescan = QPushButton("⟳")
        btn_rescan.setObjectName("iconButton")
        btn_rescan.setFixedSize(38, 38)
        btn_rescan.setToolTip(_("Rescan the faces folder"))
        btn_rescan.clicked.connect(self._reload_face_library)
        actions.addWidget(btn_rescan)

        actions.addStretch(1)

        self.ed_face_search = QLineEdit()
        self.ed_face_search.setPlaceholderText(_("Search faces..."))
        self.ed_face_search.setClearButtonEnabled(True)
        self.ed_face_search.setMaximumWidth(220)
        self.ed_face_search.textChanged.connect(self._on_face_search)
        actions.addWidget(self.ed_face_search)

        self.sw_map_faces = QCheckBox(_("Manual face mapping"))
        self.sw_map_faces.setChecked(modules.globals.map_faces)
        self.sw_map_faces.setToolTip(
            _("Assign specific source faces to specific people in the frame"))
        self.sw_map_faces.toggled.connect(self._on_map_faces_toggled)
        actions.addWidget(self.sw_map_faces)
        layout.addLayout(actions)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._faces_container = QWidget()
        self._faces_grid = QGridLayout(self._faces_container)
        self._faces_grid.setContentsMargins(0, 4, 0, 0)
        self._faces_grid.setSpacing(10)
        self._faces_grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll.setWidget(self._faces_container)
        layout.addWidget(scroll, 1)

        self.lbl_faces_empty = QLabel(
            _("No faces yet. Import a photo, or drop one onto the source panel.")
        )
        self.lbl_faces_empty.setObjectName("hint")
        layout.addWidget(self.lbl_faces_empty)
        return page

    # ── studio page (batch image / video) ────────────────────────────────

    def _build_studio_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel(_("Studio"))
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        hint = QLabel(_("Swap a face onto an image or a video file."))
        hint.setObjectName("hint")
        layout.addWidget(hint)

        frame, content = card()
        row = QHBoxLayout()
        row.setSpacing(24)

        src_col = QVBoxLayout()
        src_col.setSpacing(8)
        src_caption = QLabel(_("SOURCE FACE"))
        src_caption.setObjectName("cardTitle")
        src_col.addWidget(src_caption)
        self.studio_source = DropLabel(_("Drop a face image"))
        self.studio_source.on_file = self._apply_source_path
        src_col.addWidget(self.studio_source)
        btn_src = QPushButton(_("Select face"))
        btn_src.clicked.connect(self._on_select_source)
        src_col.addWidget(btn_src)
        row.addLayout(src_col)

        arrow = QLabel("→")
        arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(arrow)

        tgt_col = QVBoxLayout()
        tgt_col.setSpacing(8)
        tgt_caption = QLabel(_("TARGET IMAGE OR VIDEO"))
        tgt_caption.setObjectName("cardTitle")
        tgt_col.addWidget(tgt_caption)
        self.studio_target = DropLabel(_("Drop an image or video"))
        self.studio_target.on_file = self._apply_target_path
        tgt_col.addWidget(self.studio_target)
        btn_tgt = QPushButton(_("Select target"))
        btn_tgt.clicked.connect(self._on_select_target)
        tgt_col.addWidget(btn_tgt)
        row.addLayout(tgt_col)
        row.addStretch(1)
        content.addLayout(row)

        content.addWidget(separator())

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        self.btn_start = QPushButton(_("▶  Process"))
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self._on_start)
        buttons.addWidget(self.btn_start)

        btn_preview = QPushButton(_("Preview"))
        btn_preview.clicked.connect(self._on_toggle_preview)
        buttons.addWidget(btn_preview)
        buttons.addStretch(1)
        content.addLayout(buttons)

        content.addWidget(separator())
        motion_row = QHBoxLayout()
        motion_row.setSpacing(10)
        btn_motion = QPushButton(_("🎬  Motion transfer..."))
        btn_motion.setToolTip(
            _("Render a photo of someone performing your movement. Offline, "
              "minutes per clip."))
        btn_motion.clicked.connect(self._on_motion_transfer)
        motion_row.addWidget(btn_motion)
        motion_note = QLabel(_(
            "Whole body, photoreal, generated — the thing that cannot run "
            "live. Give it a photo and a clip of you moving."))
        motion_note.setObjectName("hint")
        motion_note.setWordWrap(True)
        motion_row.addWidget(motion_note, 1)
        content.addLayout(motion_row)

        layout.addWidget(frame)

        out_frame, out = card(_("Output"))
        out_grid = QGridLayout()
        out_grid.setHorizontalSpacing(24)
        out_grid.setVerticalSpacing(8)
        out_grid.addWidget(toggle(_("Keep original frame rate"), "keep_fps"), 0, 0)
        out_grid.addWidget(toggle(_("Keep audio"), "keep_audio"), 0, 1)
        out_grid.addWidget(toggle(_("Keep extracted frames"), "keep_frames",
                                  _("Leave the temporary per-frame images on disk")), 1, 0)
        out_grid.addWidget(toggle(_("Poisson blending"), "poisson_blend",
                                  _("Smoother edges, slower")), 1, 1)
        out.addLayout(out_grid)
        layout.addWidget(out_frame)
        layout.addStretch(1)
        return page

    # ── setup page ───────────────────────────────────────────────────────

    def _build_setup_page(self) -> QWidget:
        page = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)
        wrapper = QVBoxLayout(page)
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addWidget(scroll)

        layout = QVBoxLayout(inner)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel(_("Setup"))
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        # Virtual camera.
        vcam_frame, vcam = card(_("Virtual camera"))
        self.lbl_vcam_state = QLabel("")
        self.lbl_vcam_state.setWordWrap(True)
        vcam.addWidget(self.lbl_vcam_state)

        vcam_row = QHBoxLayout()
        vcam_row.setSpacing(10)
        vcam_row.addWidget(QLabel(_("Output resolution")))
        self.cb_vcam_res = QComboBox()
        for label, _w, _h in VCAM_PRESETS:
            self.cb_vcam_res.addItem(label)
        self.cb_vcam_res.setCurrentText(
            f"{modules.globals.virtual_cam_width} x {modules.globals.virtual_cam_height}"
        )
        self.cb_vcam_res.currentIndexChanged.connect(self._on_vcam_res_change)
        vcam_row.addWidget(self.cb_vcam_res)

        vcam_row.addWidget(QLabel(_("FPS")))
        self.sp_vcam_fps = QSpinBox()
        self.sp_vcam_fps.setRange(1, 120)
        self.sp_vcam_fps.setValue(modules.globals.virtual_cam_fps)
        self.sp_vcam_fps.valueChanged.connect(self._on_vcam_fps_change)
        vcam_row.addWidget(self.sp_vcam_fps)

        vcam_row.addWidget(toggle(
            _("Mirror the outgoing feed"), "virtual_cam_mirror",
            _("Flip what viewers see. Usually left off so text is readable."),
            extra=lambda v: setattr(self._vcam, "mirror", v)))
        vcam_row.addStretch(1)
        vcam.addLayout(vcam_row)

        vcam_buttons = QHBoxLayout()
        vcam_buttons.setSpacing(10)
        self.btn_install_vcam = QPushButton(_("Install virtual camera"))
        self.btn_install_vcam.clicked.connect(self._on_install_vcam)
        vcam_buttons.addWidget(self.btn_install_vcam)
        btn_recheck = QPushButton(_("Re-check"))
        btn_recheck.clicked.connect(self._refresh_vcam_availability)
        vcam_buttons.addWidget(btn_recheck)
        vcam_buttons.addStretch(1)
        vcam.addLayout(vcam_buttons)
        layout.addWidget(vcam_frame)

        # Capture.
        cap_frame, cap = card(_("Camera capture"))
        cap_row = QHBoxLayout()
        cap_row.setSpacing(10)
        cap_row.addWidget(QLabel(_("Requested resolution")))
        self.cb_cap_res = QComboBox()
        for label, _w, _h in CAPTURE_PRESETS:
            self.cb_cap_res.addItem(label)
        self.cb_cap_res.setCurrentText(
            f"{modules.globals.camera_width} x {modules.globals.camera_height}"
        )
        self.cb_cap_res.currentIndexChanged.connect(self._on_cap_res_change)
        cap_row.addWidget(self.cb_cap_res)

        cap_row.addWidget(QLabel(_("FPS")))
        self.sp_cap_fps = QSpinBox()
        self.sp_cap_fps.setRange(1, 120)
        self.sp_cap_fps.setValue(modules.globals.camera_fps)
        self.sp_cap_fps.valueChanged.connect(self._on_cap_fps_change)
        cap_row.addWidget(self.sp_cap_fps)
        cap_row.addStretch(1)
        cap.addLayout(cap_row)

        cap_note = QLabel(
            _("Cameras negotiate their own modes — the header shows what you "
              "actually got. Changes take effect the next time you go live.")
        )
        cap_note.setObjectName("hint")
        cap_note.setWordWrap(True)
        cap.addWidget(cap_note)

        cap.addWidget(toggle(
            _("Fix blueish camera tint"), "color_correction",
            _("Corrects the blue/green cast some webcams produce")))
        layout.addWidget(cap_frame)

        # Processing.
        proc_frame, proc = card(_("Processing"))
        self.lbl_provider = QLabel("")
        self.lbl_provider.setWordWrap(True)
        proc.addWidget(self.lbl_provider)
        proc.addWidget(toggle(
            _("Block explicit content"), "nsfw_filter",
            _("Refuses to process media the detector flags as explicit")))
        layout.addWidget(proc_frame)

        # Models.
        model_frame, model = card(_("Models"))
        self.lbl_models = QLabel("")
        self.lbl_models.setWordWrap(True)
        self.lbl_models.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        model.addWidget(self.lbl_models)

        model_buttons = QHBoxLayout()
        model_buttons.setSpacing(10)
        btn_move = QPushButton(_("Change folder..."))
        btn_move.setToolTip(
            _("Put the models on a drive with room. Roughly 1.5 GB."))
        btn_move.clicked.connect(self._on_change_models_dir)
        model_buttons.addWidget(btn_move)

        btn_get = QPushButton(_("Download missing"))
        btn_get.clicked.connect(self._on_download_models)
        model_buttons.addWidget(btn_get)
        model_buttons.addStretch(1)
        model.addLayout(model_buttons)
        layout.addWidget(model_frame)

        layout.addStretch(1)
        self._refresh_setup_labels()
        return page

    # ── about page ───────────────────────────────────────────────────────

    def _build_about_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel(_("About"))
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        frame, content = card()
        heading = QLabel(f"{modules.metadata.name} {modules.metadata.version}")
        heading.setObjectName("pageTitle")
        content.addWidget(heading)

        body = QLabel(
            _("{tagline}.\n\n"
              "{name} is built on the open-source {upstream} project and is "
              "released under the same licence, the {licence}. Credit for the "
              "underlying face-swap engine belongs to that project and its "
              "contributors.").format(
                  tagline=modules.metadata.tagline,
                  name=modules.metadata.name,
                  upstream=modules.metadata.upstream_name,
                  licence=modules.metadata.license_name)
        )
        body.setObjectName("hint")
        body.setWordWrap(True)
        content.addWidget(body)

        link = QLabel(
            f'<a href="{modules.metadata.upstream_url}">'
            f'{modules.metadata.upstream_url.replace("https://", "")}</a>')
        link.setObjectName("linkLabel")
        link.setOpenExternalLinks(True)
        content.addWidget(link)
        layout.addWidget(frame)

        # Keyboard shortcuts, generated from the same table that binds them.
        keys_frame, keys = card(_("Keyboard shortcuts"))
        keys_grid = QGridLayout()
        keys_grid.setHorizontalSpacing(18)
        keys_grid.setVerticalSpacing(5)
        entries = self._shortcut_table()
        half = (len(entries) + 1) // 2
        for position, (sequence, label, _slot) in enumerate(entries):
            column = 0 if position < half else 2
            row = position if position < half else position - half
            key_label = QLabel(sequence)
            key_label.setProperty("pill", True)
            key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            description = QLabel(label)
            description.setObjectName("hint")
            keys_grid.addWidget(key_label, row, column)
            keys_grid.addWidget(description, row, column + 1)
        keys_grid.setColumnStretch(1, 1)
        keys_grid.setColumnStretch(3, 1)
        keys.addLayout(keys_grid)
        layout.addWidget(keys_frame)

        # Where the app's files live.
        files_frame, files = card(_("Your files"))
        files_text = QLabel(
            f"{_('Faces')}: {FACES_DIR}\n"
            f"{_('Recordings and snapshots')}: {CAPTURES_DIR}\n"
            f"{_('Models')}: {MODELS_DIR}")
        files_text.setObjectName("hint")
        files_text.setWordWrap(True)
        files_text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        files.addWidget(files_text)
        files_row = QHBoxLayout()
        btn_captures = QPushButton(_("Open captures"))
        btn_captures.clicked.connect(self._on_open_captures)
        files_row.addWidget(btn_captures)
        btn_faces = QPushButton(_("Open faces"))
        btn_faces.clicked.connect(self._on_open_faces_folder)
        files_row.addWidget(btn_faces)
        files_row.addStretch(1)
        files.addLayout(files_row)
        layout.addWidget(files_frame)

        ethics_frame, ethics = card(_("Responsible use"))
        ethics_text = QLabel(_(
            "This software creates deepfakes. If you use a real person's face, "
            "get their consent, and label the result as synthetic when you "
            "share it. Impersonation, harassment and non-consensual imagery "
            "are illegal in many places. You are responsible for what you make "
            "with it."
        ))
        ethics_text.setObjectName("hint")
        ethics_text.setWordWrap(True)
        ethics.addWidget(ethics_text)
        layout.addWidget(ethics_frame)

        layout.addStretch(1)
        return page

    # ── live control ─────────────────────────────────────────────────────

    def _reload_cameras(self) -> None:
        indices, names = get_available_cameras()
        self._camera_indices = indices
        current = self.cb_camera.currentText() if self.cb_camera.count() else ""
        self.cb_camera.blockSignals(True)
        self.cb_camera.clear()
        if not indices:
            self.cb_camera.addItem(_("No cameras found"))
            self.cb_camera.setEnabled(False)
        else:
            self.cb_camera.addItems(names)
            self.cb_camera.setEnabled(True)
            if current in names:
                self.cb_camera.setCurrentText(current)
        self.cb_camera.blockSignals(False)
        if hasattr(self, "btn_live"):
            self.btn_live.setEnabled(bool(indices))

    def _selected_camera_index(self) -> Optional[int]:
        row = self.cb_camera.currentIndex()
        if row < 0 or row >= len(self._camera_indices):
            return None
        return self._camera_indices[row]

    def _on_live_toggled(self, checked: bool) -> None:
        if not checked:
            self._stop_live()
            return

        camera_index = self._selected_camera_index()
        if camera_index is None:
            update_status("No camera available")
            self.btn_live.setChecked(False)
            return

        if ui_dialogs.live_mapper_is_open():
            update_status("Source x Target Mapper is already open.")
            ui_dialogs.raise_live_mapper()
            self.btn_live.setChecked(False)
            return

        if modules.globals.map_faces:
            modules.globals.source_target_map = []
            ui_dialogs.open_live_mapper_dialog(
                camera_index, modules.globals.source_target_map)
            self.btn_live.setChecked(False)
            return

        if not modules.globals.source_path:
            update_status("Select a source face first")
            self.btn_live.setChecked(False)
            self._pages.setCurrentIndex(FACES_PAGE_INDEX)
            self._nav_group.button(FACES_PAGE_INDEX).setChecked(True)
            return

        self._start_live(camera_index)

    def _start_live(self, camera_index: int) -> None:
        """Warm the models, then start the capture and processing threads."""
        # An installed build ships without the models; fetch them once here
        # rather than letting the swap fail with nothing on screen.
        from modules import ui_first_run
        if not ui_first_run.run_if_needed(self):
            update_status("The face swap models are still missing.")
            self.btn_live.setChecked(False)
            self._refresh_setup_labels()
            return
        self._refresh_setup_labels()

        update_status("Loading models...")
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        try:
            from modules.face_analyser import get_face_analyser
            from modules.processors.frame.face_swapper import get_face_swapper
            get_face_analyser()
            get_face_swapper()
        except Exception as exc:
            update_status(f"Could not load models: {exc}")
            self.btn_live.setChecked(False)
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._engine.add_sink(self._preview_sink)
        started = self._engine.start(
            camera_index,
            modules.globals.camera_width,
            modules.globals.camera_height,
            modules.globals.camera_fps,
        )
        if not started:
            self._engine.remove_sink(self._preview_sink)
            update_status(self._engine.error or "Failed to start the camera")
            self.btn_live.setChecked(False)
            return

        self.btn_live.setChecked(True)
        self.btn_live.setText(_("■  Stop"))
        self._live_started_at = time.time()
        width, height = self._engine.camera_resolution
        update_status(
            f"Live on {width}x{height} at {self._engine.camera_fps:.0f} fps"
        )

        # Auto-resume the virtual camera if it was on last session.
        if modules.globals.virtual_cam_enabled and not self._vcam.is_running:
            self.btn_vcam.setChecked(True)
            self._on_vcam_toggled(True)

    def _stop_live(self) -> None:
        # Close the recording first so the file is finished with frames that
        # were actually captured, rather than being cut off mid-write.
        if self._recorder.is_recording:
            self.btn_record.setChecked(False)
            self._on_record_toggled(False)
        self._engine.stop()
        self._engine.remove_sink(self._preview_sink)
        if self._vcam.is_running:
            self._vcam.stop()
            self._engine.remove_sink(self._vcam)
            self.btn_vcam.setChecked(False)
        self.btn_live.setChecked(False)
        self.btn_live.setText(_("▶  Go Live"))
        self._live_started_at = 0.0
        self.video_surface.clear()
        self.video_surface.setText(_("Stopped."))
        update_status("Stopped")

    # ── virtual camera ───────────────────────────────────────────────────

    def _refresh_vcam_availability(self) -> None:
        available, reason = VirtualCamSink.available()
        self._vcam_available = available
        self.btn_vcam.setEnabled(available)
        self.lbl_vcam_hint.setText("" if available else reason)
        if hasattr(self, "lbl_vcam_state"):
            self.lbl_vcam_state.setText(reason)
            self.btn_install_vcam.setVisible(not available)

    def _on_vcam_toggled(self, checked: bool) -> None:
        if not checked:
            self._engine.remove_sink(self._vcam)
            self._vcam.stop()
            modules.globals.virtual_cam_enabled = False
            save_switch_states()
            update_status("Virtual camera stopped")
            return

        self._vcam.mirror = modules.globals.virtual_cam_mirror
        started = self._vcam.start(
            modules.globals.virtual_cam_width,
            modules.globals.virtual_cam_height,
            modules.globals.virtual_cam_fps,
        )
        if not started:
            self.btn_vcam.setChecked(False)
            self.lbl_vcam_hint.setText(self._vcam.error)
            update_status(self._vcam.error or "Could not start the virtual camera")
            return

        self._engine.add_sink(self._vcam)
        modules.globals.virtual_cam_enabled = True
        save_switch_states()
        width, height = self._vcam.resolution
        name = self._vcam.device_name or "virtual camera"
        self.lbl_vcam_hint.setText(
            _("Publishing {w}x{h} to \"{name}\" — select it as your camera in "
              "Discord, Zoom, Teams or your browser.").format(
                  w=width, h=height, name=name)
        )
        update_status(f"Virtual camera live as \"{name}\"")

    def _on_install_vcam(self) -> None:
        answer = QMessageBox.question(
            self,
            _("Install virtual camera"),
            _("{app} publishes its feed through the OBS Studio virtual "
              "camera driver, which has to be installed once. OBS never needs "
              "to be running.\n\nOpen the OBS Studio download page in your "
              "browser?"),
            QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Open:
            webbrowser.open(OBS_DOWNLOAD_URL)
            update_status(
                "After installing OBS, come back and press Re-check.")

    def _on_vcam_res_change(self, index: int) -> None:
        if 0 <= index < len(VCAM_PRESETS):
            _label, width, height = VCAM_PRESETS[index]
            modules.globals.virtual_cam_width = width
            modules.globals.virtual_cam_height = height
            save_switch_states()
            if self._vcam.is_running:
                update_status(
                    "Virtual camera resolution applies when you restart it.")

    def _on_vcam_fps_change(self, value: int) -> None:
        modules.globals.virtual_cam_fps = value
        save_switch_states()

    def _on_cap_res_change(self, index: int) -> None:
        if 0 <= index < len(CAPTURE_PRESETS):
            _label, width, height = CAPTURE_PRESETS[index]
            modules.globals.camera_width = width
            modules.globals.camera_height = height
            if (width, height) != PERFORMANCE_PRESETS.get(
                    modules.globals.performance_preset, {}).get("camera"):
                self._mark_custom_preset()
            save_switch_states()

    def _on_cap_fps_change(self, value: int) -> None:
        modules.globals.camera_fps = value
        save_switch_states()

    # ── source faces ─────────────────────────────────────────────────────

    def _apply_source_path(self, path: str) -> None:
        if not path or not is_image(path):
            update_status("That file is not an image")
            return
        modules.globals.source_path = path
        RECENT_DIRS["source"] = os.path.dirname(path)
        self._sync_source_previews()
        self._highlight_selected_face()
        self._describe_identity(path)
        self._warn_if_no_face(path)

    def _warn_if_no_face(self, path: str) -> None:
        """Say so up front when a source image is a poor choice.

        Without this the swap just silently does nothing (no face) or comes out
        vaguely wrong (face too small), which reads as the app being broken.
        Skipped before the models are loaded so choosing a face never triggers a
        multi-second stall on a cold start.

        The size check is free: detection has already run by this point, so the
        bounding box is sitting right there. It is worth saying because it
        cannot be fixed later — see modules/face_quality.py for why upscaling a
        small source face measurably makes the likeness worse, not better.
        """
        import modules.face_analyser as face_analyser
        if face_analyser.FACE_ANALYSER is None:
            return
        from modules import face_quality, imread_unicode
        image = imread_unicode(path)
        if image is None:
            update_status(f"Could not read {os.path.basename(path)}")
            return
        face = face_analyser.get_source_face(image)
        if face is None:
            update_status(
                f"No face found in {os.path.basename(path)} — try a photo "
                "with the whole head visible.")
            return
        message = face_quality.describe(
            os.path.basename(path), face_quality.face_width(face))
        if message:
            update_status(message)

    def _apply_target_path(self, path: str) -> None:
        if not path:
            return
        if is_image(path):
            pixmap = render_image_preview(path, (DROP_PREVIEW_SIZE, DROP_PREVIEW_SIZE))
        elif is_video(path):
            pixmap = render_video_preview(path, (DROP_PREVIEW_SIZE, DROP_PREVIEW_SIZE))
        else:
            update_status("Unsupported target file")
            return
        modules.globals.target_path = path
        RECENT_DIRS["target"] = os.path.dirname(path)
        if pixmap:
            self.studio_target.setPixmap(pixmap)
            self.studio_target.setText("")
        update_status(f"Target: {os.path.basename(path)}")

    def _sync_source_previews(self) -> None:
        path = modules.globals.source_path
        if path and os.path.isfile(path):
            self.sidebar_face.setPixmap(
                render_image_preview(path, (SIDEBAR_WIDTH - 32, SIDEBAR_WIDTH - 32)))
            self.sidebar_face.setText("")
            self.studio_source.setPixmap(
                render_image_preview(path, (DROP_PREVIEW_SIZE, DROP_PREVIEW_SIZE)))
            self.studio_source.setText("")
        else:
            self.sidebar_face.clear_image()
            self.studio_source.clear_image()

    def _on_select_source(self) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("Select a source face"), RECENT_DIRS["source"] or "",
            IMAGE_FILTER)
        if path:
            self._apply_source_path(path)

    def _on_select_target(self) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("Select a target image or video"),
            RECENT_DIRS["target"] or "", MEDIA_FILTER)
        if path:
            self._apply_target_path(path)

    def _on_import_faces(self) -> None:
        paths, _f = QFileDialog.getOpenFileNames(
            self, _("Import face images"), RECENT_DIRS["source"] or "",
            IMAGE_FILTER)
        if not paths:
            return
        ensure_user_dirs()
        imported = 0
        for path in paths:
            destination = os.path.join(FACES_DIR, os.path.basename(path))
            if os.path.abspath(destination) == os.path.abspath(path):
                imported += 1
                continue
            try:
                shutil.copy2(path, destination)
                imported += 1
            except OSError as exc:
                update_status(f"Could not import {os.path.basename(path)}: {exc}")
        RECENT_DIRS["source"] = os.path.dirname(paths[0])
        self._reload_face_library()
        if imported:
            self._apply_source_path(
                os.path.join(FACES_DIR, os.path.basename(paths[0])))
        update_status(f"Imported {imported} face(s)")

    def _on_open_faces_folder(self) -> None:
        ensure_user_dirs()
        _open_folder(FACES_DIR)

    def _on_random_face(self) -> None:
        try:
            response = requests.get(
                RANDOM_FACE_URL,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            update_status(f"Could not fetch a random face: {exc}")
            return

        # The site used to serve the image from its root and now serves an
        # HTML page there. Check we actually got an image so a future change
        # surfaces as a message instead of a corrupt file.
        if not response.content.startswith(b"\xff\xd8\xff"):
            update_status(
                "The random face service returned something that is not a "
                "JPEG. Import a face image instead.")
            return

        # A unique name per fetch. Reusing one filename meant the live loop,
        # which caches the source face, could not tell that the picture had
        # changed — the thumbnail updated but the swap did not.
        ensure_user_dirs()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(FACES_DIR, f"random-{stamp}-{os.getpid() % 1000}.jpg")
        try:
            with open(path, "wb") as handle:
                handle.write(response.content)
        except OSError as exc:
            update_status(f"Could not save the random face: {exc}")
            return
        self._reload_face_library()
        self._apply_source_path(path)

    # ── face library ─────────────────────────────────────────────────────

    FACE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif")

    def _favourites(self) -> set:
        stored = paths.read_config().get("favourite_faces", [])
        return {os.path.abspath(p) for p in stored} if isinstance(stored, list) else set()

    def _set_favourite(self, path: str, favourite: bool) -> None:
        current = self._favourites()
        if favourite:
            current.add(os.path.abspath(path))
        else:
            current.discard(os.path.abspath(path))
        config = paths.read_config()
        config["favourite_faces"] = sorted(current)
        paths.write_config(config)
        self._reload_face_library()

    def _on_face_search(self, text: str) -> None:
        self._face_filter = text.strip().lower()
        self._reload_face_library()

    def _scan_faces(self) -> List[str]:
        """Face files, favourites first, then alphabetical; filtered by search."""
        if not os.path.isdir(FACES_DIR):
            return []
        try:
            names = os.listdir(FACES_DIR)
        except OSError:
            return []
        found = [
            os.path.join(FACES_DIR, name) for name in names
            if name.lower().endswith(self.FACE_EXTENSIONS)
            and (not self._face_filter or self._face_filter in name.lower())
        ]
        favourites = self._favourites()
        return sorted(
            found,
            key=lambda p: (os.path.abspath(p) not in favourites,
                           os.path.basename(p).lower()),
        )

    def _reload_face_library(self) -> None:
        while self._faces_grid.count():
            item = self._faces_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._face_buttons = []

        files = self._scan_faces()
        self._face_paths = [os.path.abspath(p) for p in files]
        favourites = self._favourites()

        from modules import face_identity
        group_sizes: dict = {}
        for path in files:
            group_sizes.setdefault(face_identity.group_key(path), []).append(path)

        if not files:
            self.lbl_faces_empty.setVisible(True)
            self.lbl_faces_empty.setText(
                _("Nothing matches \"{q}\".").format(q=self._face_filter)
                if self._face_filter else
                _("No faces yet. Use Find faces, or drop a photo onto the "
                  "source panel."))
        else:
            self.lbl_faces_empty.setVisible(False)

        columns = 7
        for index, path in enumerate(files):
            starred = os.path.abspath(path) in favourites
            button = QPushButton()
            button.setObjectName("faceTile")
            button.setCheckable(True)
            button.setFixedSize(FACE_THUMB_SIZE, FACE_THUMB_SIZE)
            button.setIconSize(QSize(FACE_THUMB_SIZE - 8, FACE_THUMB_SIZE - 8))
            tip = ("★ " if starred else "") + os.path.basename(path)
            group_size = len(group_sizes.get(face_identity.group_key(path), ()))
            if modules.globals.blend_identity and group_size > 1:
                tip += "\n" + _("{n} photos blended as one identity").format(
                    n=group_size)
            button.setToolTip(tip)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setProperty("starred", starred)
            try:
                pixmap = render_image_preview(
                    path, (FACE_THUMB_SIZE - 8, FACE_THUMB_SIZE - 8))
                button.setIcon(QIcon(pixmap))
            except Exception:
                button.setText(os.path.basename(path)[:10])
            button.clicked.connect(lambda _c, p=path: self._apply_source_path(p))
            button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            button.customContextMenuRequested.connect(
                lambda pos, p=path, b=button: self._face_context_menu(p, b, pos))
            button._face_path = path
            self._faces_grid.addWidget(button, index // columns, index % columns)
            self._face_buttons.append(button)

        self._highlight_selected_face()

    def _face_context_menu(self, path: str, button: QPushButton, pos) -> None:
        favourites = self._favourites()
        starred = os.path.abspath(path) in favourites

        menu = QMenu(self)
        act_use = menu.addAction(_("Use this face"))
        act_star = menu.addAction(
            _("Remove star") if starred else _("Star"))
        menu.addSeparator()
        act_rename = menu.addAction(_("Rename..."))
        act_reveal = menu.addAction(_("Show in folder"))
        menu.addSeparator()
        act_delete = menu.addAction(_("Delete"))

        chosen = menu.exec(button.mapToGlobal(pos))
        if chosen is act_use:
            self._apply_source_path(path)
        elif chosen is act_star:
            self._set_favourite(path, not starred)
        elif chosen is act_rename:
            self._rename_face(path)
        elif chosen is act_reveal:
            _open_folder(FACES_DIR)
        elif chosen is act_delete:
            self._delete_face(path)

    def _rename_face(self, path: str) -> None:
        stem, extension = os.path.splitext(os.path.basename(path))
        new_stem, ok = QInputDialog.getText(
            self, _("Rename face"), _("New name:"), text=stem)
        new_stem = (new_stem or "").strip()
        if not ok or not new_stem or new_stem == stem:
            return
        # Keep it a plain filename — no traversal, no directory separators.
        safe = "".join(c for c in new_stem if c not in '\\/:*?"<>|').strip()
        if not safe:
            update_status("That name cannot be used for a file.")
            return
        destination = os.path.join(FACES_DIR, safe + extension)
        if os.path.exists(destination):
            update_status(f"{safe}{extension} already exists.")
            return
        try:
            was_source = (modules.globals.source_path
                          and os.path.abspath(modules.globals.source_path)
                          == os.path.abspath(path))
            was_favourite = os.path.abspath(path) in self._favourites()
            os.rename(path, destination)
            if was_favourite:
                self._set_favourite(path, False)
                self._set_favourite(destination, True)
            if was_source:
                modules.globals.source_path = destination
            self._reload_face_library()
            self._sync_source_previews()
            update_status(f"Renamed to {safe}{extension}")
        except OSError as exc:
            update_status(f"Could not rename: {exc}")

    def _delete_face(self, path: str) -> None:
        answer = QMessageBox.question(
            self, _("Delete face"),
            _("Delete {name} from your library?\n\nThis removes the file "
              "from disk.").format(name=os.path.basename(path)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            os.remove(path)
        except OSError as exc:
            update_status(f"Could not delete: {exc}")
            return
        if (modules.globals.source_path
                and os.path.abspath(modules.globals.source_path)
                == os.path.abspath(path)):
            modules.globals.source_path = None
            self._sync_source_previews()
        self._set_favourite(path, False)
        self._reload_face_library()
        update_status(f"Deleted {os.path.basename(path)}")

    def _on_find_faces(self) -> None:
        from modules.ui_face_browser import FaceBrowserDialog

        dialog = FaceBrowserDialog(self)
        dialog.exec()
        if dialog.imported:
            self._reload_face_library()
            self._apply_source_path(dialog.imported[0])

    def _highlight_selected_face(self) -> None:
        current = modules.globals.source_path
        for button in self._face_buttons:
            match = bool(current) and os.path.abspath(
                button._face_path) == os.path.abspath(current)
            button.setChecked(match)

    # ── processing options ───────────────────────────────────────────────

    @staticmethod
    def _current_enhancer() -> str:
        for label, key in ENHANCER_KEYS.items():
            if key and modules.globals.fp_ui.get(key, False):
                return label
        return "None"

    def _on_enhancer_change(self, choice: str) -> None:
        for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
            modules.globals.fp_ui[key] = False
        selected = ENHANCER_KEYS.get(choice)
        if selected:
            modules.globals.fp_ui[selected] = True
        if choice != PERFORMANCE_PRESETS.get(
                modules.globals.performance_preset, {}).get("enhancer"):
            self._mark_custom_preset()
        save_switch_states()
        # Rebuild the chain so the change lands on the next live frame.
        get_frame_processors_modules(modules.globals.frame_processors)
        self._engine.request_processor_reload()

    def _on_blend_changed(self) -> None:
        from modules import face_identity
        face_identity.clear_cache()
        # Force the live loop to rebuild the source on its next frame.
        self._engine.request_processor_reload()
        if modules.globals.source_path:
            self._describe_identity(modules.globals.source_path)

    def _describe_identity(self, path: str) -> None:
        """Say how many photos are behind the current source face."""
        from modules import face_identity
        if not modules.globals.blend_identity:
            update_status(f"Source face: {os.path.basename(path)}")
            return
        group = face_identity.find_group(path)
        if len(group) > 1:
            update_status(
                f"Source: {face_identity.group_key(path)} "
                f"- blending {len(group)} photos")
        else:
            update_status(f"Source face: {os.path.basename(path)}")

    def _on_preset_change(self, name: str) -> None:
        preset = PERFORMANCE_PRESETS.get(name)
        if preset is None:
            return
        modules.globals.performance_preset = name
        if preset["camera"] is not None:
            modules.globals.camera_width, modules.globals.camera_height = preset["camera"]
            modules.globals.detect_interval_ratio = preset["detect_ratio"]
            if hasattr(self, "cb_cap_res"):
                self.cb_cap_res.blockSignals(True)
                self.cb_cap_res.setCurrentText(
                    f"{preset['camera'][0]} x {preset['camera'][1]}")
                self.cb_cap_res.blockSignals(False)
            self.cb_enhancer.setCurrentText(preset["enhancer"])
        save_switch_states()
        self._engine.request_processor_reload()
        suffix = _(" Capture size applies next time you go live.") \
            if self._engine.is_running and preset["camera"] else ""
        update_status(f"{name}: {preset['hint']}{suffix}")

    def _mark_custom_preset(self) -> None:
        """Drop to Custom when a setting no longer matches the chosen preset."""
        if modules.globals.performance_preset == "Custom":
            return
        modules.globals.performance_preset = "Custom"
        if hasattr(self, "cb_preset"):
            self.cb_preset.blockSignals(True)
            self.cb_preset.setCurrentText("Custom")
            self.cb_preset.blockSignals(False)

    def _on_opacity_change(self, value: float) -> None:
        modules.globals.opacity = value
        modules.globals.face_swapper_enabled = value > 0

    def _on_sharpness_change(self, value: float) -> None:
        modules.globals.sharpness = value

    def _on_likeness_change(self, value: float) -> None:
        modules.globals.identity_strength = value

    def _on_mouth_mask_change(self, value: float) -> None:
        modules.globals.mouth_mask_size = value
        modules.globals.mouth_mask = value > 0
        if value <= 0:
            modules.globals.show_mouth_mask_box = False

    def _on_mouth_mask_pressed(self) -> None:
        if modules.globals.mouth_mask_size > 0:
            modules.globals.show_mouth_mask_box = True

    def _on_mouth_mask_released(self) -> None:
        modules.globals.show_mouth_mask_box = False

    def _on_map_faces_toggled(self, value: bool) -> None:
        modules.globals.map_faces = value
        save_switch_states()
        if not value:
            ui_dialogs.close_mapper_window()

    # ── studio actions ───────────────────────────────────────────────────

    def _on_start(self) -> None:
        if ui_dialogs.mapper_is_open():
            update_status("Please complete pop-up or close it.")
            return
        if not modules.globals.target_path:
            update_status("Select a target image or video first")
            return
        if not modules.globals.map_faces and not modules.globals.source_path:
            update_status("Select a source face first")
            return

        if modules.globals.map_faces:
            modules.globals.source_target_map = []
            if is_image(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_image()
            elif is_video(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_video()
            if modules.globals.source_target_map:
                ui_dialogs.open_mapper_dialog(
                    self._start_cb, modules.globals.source_target_map)
            else:
                update_status("No faces found in target")
        else:
            self.select_output_and_start()

    def select_output_and_start(self) -> None:
        """Ask where to save, then run the batch pipeline."""
        if is_image(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("Save image output"),
                os.path.join(RECENT_DIRS["output"] or "", "output.png"),
                "Images (*.png *.jpg *.jpeg *.bmp)",
            )
        elif is_video(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("Save video output"),
                os.path.join(RECENT_DIRS["output"] or "", "output.mp4"),
                "Videos (*.mp4 *.mkv)",
            )
        else:
            return
        if not path:
            return
        modules.globals.output_path = path
        RECENT_DIRS["output"] = os.path.dirname(path)
        self.btn_start.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        try:
            self._start_cb()
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_start.setEnabled(True)

    def _build_motion_page(self) -> QWidget:
        from modules.ui_motion import MotionTransferPage

        self._motion_page = MotionTransferPage(self)
        return self._motion_page

    def _on_motion_transfer(self) -> None:
        """The Studio shortcut just moves to the page it used to pop open."""
        self._pages.setCurrentIndex(MOTION_PAGE_INDEX)
        button = self._nav_group.button(MOTION_PAGE_INDEX)
        if button is not None:
            button.setChecked(True)

    def _on_toggle_preview(self) -> None:
        if not (modules.globals.source_path and modules.globals.target_path):
            update_status("Select a source face and a target first")
            return
        if self._preview is None:
            self._preview = ui_dialogs.PreviewWindow()
        if self._preview.isVisible():
            self._preview.hide()
            return
        self._preview.init_for_target()
        self._preview.refresh_frame(0)
        self._preview.show()

    # ── recording and snapshots ──────────────────────────────────────────

    def _on_record_toggled(self, checked: bool) -> None:
        if not checked:
            path = self._recorder.stop()
            self._engine.remove_sink(self._recorder)
            self.btn_record.setText(_("●  Record"))
            if path and os.path.isfile(path):
                size = os.path.getsize(path) / 1_048_576
                update_status(
                    f"Saved {os.path.basename(path)} "
                    f"({self._recorder.frames} frames, {size:.0f} MB)")
            return

        if not self._engine.is_running:
            update_status("Go live before recording")
            self.btn_record.setChecked(False)
            return

        frame = self._preview_sink.get()
        if frame is None:
            update_status("Waiting for the first frame — try again in a moment")
            self.btn_record.setChecked(False)
            return

        ensure_user_dirs()
        path = recorder.default_output_path(CAPTURES_DIR)
        height, width = frame.shape[:2]
        fps = self._engine.fps or self._engine.camera_fps or 30.0
        if not self._recorder.start(path, width, height, fps):
            update_status(self._recorder.error or "Could not start recording")
            self.btn_record.setChecked(False)
            return

        self._engine.add_sink(self._recorder)
        self.btn_record.setText(_("■  Stop rec"))
        update_status(f"Recording to {os.path.basename(path)}")

    def _on_snapshot(self) -> None:
        frame = self._preview_sink.get()
        if frame is None:
            update_status("Nothing to capture — go live first")
            return
        ensure_user_dirs()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(CAPTURES_DIR, f"morphify-{stamp}.png")
        if imwrite_unicode(path, frame):
            update_status(f"Saved {os.path.basename(path)}")
        else:
            update_status("Could not save the snapshot")

    def _on_open_captures(self) -> None:
        ensure_user_dirs()
        _open_folder(CAPTURES_DIR)

    # ── models ───────────────────────────────────────────────────────────

    def _on_change_models_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, _("Choose a folder for the models"), MODELS_DIR)
        if not chosen:
            return

        free = paths.free_bytes(chosen)
        if 0 <= free < paths.MODELS_REQUIRED_BYTES:
            QMessageBox.warning(
                self, _("Not enough space"),
                _("That drive has {free:.1f} GB free. The models need about "
                  "{need:.1f} GB.").format(
                      free=free / 1024 ** 3,
                      need=paths.MODELS_REQUIRED_BYTES / 1024 ** 3),
            )
            return

        if not paths.set_models_dir(chosen):
            update_status("Could not save the models folder setting.")
            return

        QMessageBox.information(
            self, _("Models folder changed"),
            _("Models will be read from:\n\n{path}\n\nRestart {app} for this "
              "to take effect.").format(
                  path=chosen, app=modules.metadata.name),
        )
        self._refresh_setup_labels()

    def _on_download_models(self) -> None:
        from modules import model_store, ui_first_run

        shortfall = model_store.space_shortfall()
        if shortfall > 0:
            QMessageBox.warning(
                self, _("Not enough space"),
                _("Downloading the missing models needs about {need:.1f} GB "
                  "more room on:\n\n{path}\n\nUse \"Change folder...\" to put "
                  "them on another drive.").format(
                      need=shortfall / 1024 ** 3, path=MODELS_DIR),
            )
            return

        pending = model_store.missing_models()
        if not pending:
            update_status("All models are already downloaded.")
            return
        dialog = ui_first_run.ModelDownloadDialog(pending, self)
        dialog.start()
        dialog.exec()
        self._refresh_setup_labels()

    # ── periodic refresh ─────────────────────────────────────────────────

    def _tick(self) -> None:
        if self._shutting_down:
            return

        # The camera can drop out on its own (unplugged, taken by another
        # app); reflect that instead of showing a frozen last frame.
        if self.btn_live.isChecked() and self._engine.has_failed:
            update_status("The camera stopped responding")
            self._stop_live()
            return

        frame = self._preview_sink.take()
        if frame is not None:
            if modules.globals.split_view:
                frame = compose_split(self._engine.last_original, frame)
            fitted = fit_image_to_size(
                frame, self.video_surface.width(), self.video_surface.height())
            self.video_surface.setPixmap(bgr_to_qpixmap(fitted))

        self._refresh_pills()
        self._refresh_session_label()

    def _refresh_session_label(self) -> None:
        parts = []
        if self._engine.is_running and self._live_started_at:
            parts.append(_("Live {t}").format(
                t=format_duration(time.time() - self._live_started_at)))
        if self._recorder.is_recording:
            line = _("REC {t}").format(
                t=format_duration(self._recorder.elapsed))
            if self._recorder.dropped:
                line += _(" ({n} dropped)").format(n=self._recorder.dropped)
            parts.append(line)
        self.lbl_session.setText("     ".join(parts))

    def _refresh_pills(self) -> None:
        if self._engine.is_running:
            width, height = self._engine.camera_resolution
            camera_fps = self._engine.camera_fps
            self.pill_device.set(f"{width}×{height}", "busy")
            state = "warn" if modules.globals.bypass_swap else "busy"
            label = (_("BYPASS") if modules.globals.bypass_swap
                     else f"{self._engine.fps:.0f} FPS")
            self.pill_fps.set(label, state)

            # Distinguish "the swap is slow" from "the camera is only
            # sending this many frames". Webcams halve their rate in low
            # light, which otherwise reads as the app being at fault.
            hint = _("Swap loop: {loop:.0f} fps\nCamera delivers: {cam:.0f} fps"
                     ).format(loop=self._engine.fps, cam=camera_fps)
            if camera_fps and self._engine.fps >= camera_fps * 0.9:
                hint += _("\n\nYou are camera-limited, not GPU-limited. "
                          "Webcams drop to half rate in low light — more "
                          "light usually doubles this.")
            self.pill_fps.setToolTip(hint)
        else:
            self.pill_device.set(_("IDLE"), "idle")
            self.pill_fps.set("— FPS", "idle")
            self.pill_fps.setToolTip("")

        if self._recorder.is_recording:
            self.pill_vcam.set("● REC", "error")
        elif self._vcam.is_running:
            self.pill_vcam.set("VCAM ON", "live")
        elif not getattr(self, "_vcam_available", True):
            self.pill_vcam.set("VCAM N/A", "warn")
        else:
            self.pill_vcam.set("VCAM OFF", "idle")

    def _refresh_setup_labels(self) -> None:
        providers = modules.globals.execution_providers or ["(not set)"]
        primary = providers[0].replace("ExecutionProvider", "")
        self.lbl_provider.setText(
            _("Running on {primary}. Threads: {threads}.").format(
                primary=primary,
                threads=modules.globals.execution_threads or "auto")
        )

        free = paths.free_bytes(MODELS_DIR)
        space = (f"  —  {free / 1024 ** 3:.1f} GB free" if free >= 0 else "")
        lines = [f"{_('Folder')}: {MODELS_DIR}{space}"]
        if 0 <= free < paths.MODELS_REQUIRED_BYTES:
            lines.append(
                _("  ! This drive is short on space. The full model set needs "
                  "about {need:.1f} GB — use \"Change folder...\" below.")
                .format(need=paths.MODELS_REQUIRED_BYTES / 1024 ** 3))
        expected = [
            "inswapper_128_fp16.onnx",
            "inswapper_128.onnx",
            "GFPGANv1.4.onnx",
            "GPEN-BFR-512.onnx",
            "GPEN-BFR-256.onnx",
        ]
        for name in expected:
            path = os.path.join(MODELS_DIR, name)
            if os.path.isfile(path):
                size = os.path.getsize(path) / 1_048_576
                lines.append(f"  ✓ {name}  ({size:.0f} MB)")
            else:
                lines.append(f"  ✗ {name}  {_('missing')}")
        self.lbl_models.setText("\n".join(lines))

    def set_status(self, text: str) -> None:
        self.pill_status.setText(text)

    # ── shutdown ─────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        # Closing the window while live usually means "get this off my
        # screen", not "drop the feed my call is using". Hide to the tray
        # instead, and let the tray's Quit actually exit.
        if (not getattr(self, "_force_quit", False)
                and self._tray is not None
                and (self._engine.is_running or self._vcam.is_running)):
            event.ignore()
            self.hide()
            self._tray.showMessage(
                modules.metadata.name,
                _("Still running — the feed is live. Use the tray icon to "
                  "come back or quit."),
                QSystemTrayIcon.MessageIcon.Information, 4000)
            return

        self._shutting_down = True
        self._ui_timer.stop()
        try:
            self._recorder.stop()
        except Exception:
            pass
        try:
            self._vcam.stop()
        except Exception:
            pass
        try:
            self._engine.stop()
        except Exception:
            pass
        if self._preview is not None:
            self._preview.close()
        if self._tray is not None:
            self._tray.hide()
        ui_dialogs.close_mapper_window()
        save_switch_states()
        event.accept()
        self._destroy_cb()


# ─── entry point ─────────────────────────────────────────────────────────


class _Window:
    """Thin wrapper exposing .mainloop() for core.py compatibility."""

    def __init__(self, app: QApplication, main_window: MainWindow):
        self._app = app
        self._main = main_window

    def mainloop(self) -> None:
        self._main.show()
        self._app.exec()


def init(start: Callable[[], None], destroy: Callable[[], None],
         lang: str) -> _Window:
    global _APP, _MAIN

    set_language(lang)
    _APP = QApplication.instance() or QApplication(sys.argv)
    _APP.setStyleSheet(stylesheet())
    _APP.setApplicationName(modules.metadata.name)
    _APP.setApplicationDisplayName(modules.metadata.name)
    icon_path = app_icon_path()
    if icon_path:
        _APP.setWindowIcon(QIcon(icon_path))
    # Hiding to the tray must not end the process.
    _APP.setQuitOnLastWindowClosed(False)

    _MAIN = MainWindow(start, destroy)
    init_bridge(_APP, _MAIN.set_status)
    ui_dialogs.bind_main_window(_MAIN, _MAIN._start_live)

    return _Window(_APP, _MAIN)
