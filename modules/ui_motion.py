"""The Motion page: Wan-Animate-2 motion transfer, built into the main window.

This was a separate dialog once, which was a mistake worth recording. A render
here runs for tens of minutes, and a floating window invites you to close it --
which cancelled the job and discarded every finished pass. As a page it cannot
be closed by accident, it keeps reporting while you use the rest of the app, and
stopping is something you have to actually mean.

A job is expensive, so the page spends effort up front on telling you what a set
of choices will cost before you commit, and on refusing to start when something
is plainly missing.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from typing import Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from modules import wan_animate as wa
from modules.paths import CAPTURES_DIR, USER_DATA_DIR, ensure_user_dirs
from modules.ui_common import RECENT_DIRS, _, render_image_preview, update_status

VIDEO_FILTER = "Videos (*.mp4 *.mkv *.mov *.avi *.webm)"
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp)"

LOG_PATH = os.path.join(USER_DATA_DIR, "motion-transfer.log")

# Pixel budgets, not fixed shapes: the aspect ratio comes from your clip, so a
# landscape video does not get squeezed into a portrait frame.
SIZES = [
    (_("Auto - match my video"), 480 * 848),
    (_("Smaller - about 30% faster"), 400 * 704),
    (_("Larger - sharper, much slower"), 576 * 1024),
]

QUALITY = [
    (_("Draft (6 steps)"), 6),
    (_("Normal (10 steps) - the tuned default"), 10),
    (_("Good (16 steps)"), 16),
    (_("Best (24 steps)"), 24),
]

CACHE = [
    (_("No cache - safest on 24 GB"), "cpu", wa.CACHE_OFF),
    (_("Cached, int4 - faster, needs ~3 GB more"), "cpu", "int4"),
    (_("Cached, int8 - faster, needs ~6 GB more"), "cpu", "int8"),
]

#: Presets live in the engine; the page only translates their labels.
SMOOTHNESS = [(_(label), gen, out, saving)
              for label, gen, out, saving in wa.SMOOTHNESS_PRESETS]


class _Signals(QObject):
    progress = Signal(str, float)
    finished = Signal(object, str)   # TransferResult or None, error


class MotionTransferPage(QWidget):
    """Reference photo + a video of you -> that person doing your motion."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signals = _Signals()
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._backend = wa.Backend()
        self._reference = ""
        self._driving = ""
        self._clip: Optional[tuple[int, float, int, int]] = None
        self._started_at = 0.0
        self._last_output = ""

        # The page is denser than the window is tall on a 1080p screen. Without
        # a scroll area Qt does not clip -- it *compresses*, squashing the top
        # rows until their buttons read "hoose" and fields overlap the row
        # below. Scrolling keeps every control at its natural size.
        page = QVBoxLayout(self)
        page.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        scroll.setWidget(body)
        page.addWidget(scroll)

        outer = QVBoxLayout(body)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(12)

        title = QLabel(_("Motion transfer"))
        title.setObjectName("pageTitle")
        outer.addWidget(title)

        intro = QLabel(_(
            "Give it a photo of someone and a video of you moving. Wan-Animate-2 "
            "regenerates that person performing your motion — whole body, face "
            "and hands, from the raw video rather than a traced skeleton. Clips "
            "of any length work; it chains passes automatically."))
        intro.setObjectName("hint")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        columns = QHBoxLayout()
        columns.setSpacing(18)
        columns.addLayout(self._build_inputs(), 1)
        columns.addLayout(self._build_prompts(), 1)
        outer.addLayout(columns)

        outer.addWidget(self._separator())

        self._estimate = QLabel("")
        self._estimate.setObjectName("hint")
        self._estimate.setWordWrap(True)
        outer.addWidget(self._estimate)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%p%")
        outer.addWidget(self._bar)

        self._status = QLabel("")
        self._status.setObjectName("hint")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        buttons = QHBoxLayout()
        self._btn_start = QPushButton(_("Render"))
        self._btn_start.setObjectName("primary")
        self._btn_start.clicked.connect(self._on_start)
        buttons.addWidget(self._btn_start)
        self._btn_stop = QPushButton(_("Stop"))
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        buttons.addWidget(self._btn_stop)
        btn_open = QPushButton(_("Open output folder"))
        btn_open.clicked.connect(self._open_output)
        buttons.addWidget(btn_open)
        buttons.addStretch(1)
        outer.addLayout(buttons)
        outer.addStretch(1)

        self._signals.progress.connect(self._on_progress)
        self._signals.finished.connect(self._on_finished)
        self._update_estimate()
        self._check_ready()

    # ── layout helpers ───────────────────────────────────────────────────

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setObjectName("separator")
        return line

    def _build_inputs(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(8)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        # Only the field column may absorb slack. Without this the browse
        # buttons get squeezed until their label is clipped mid-word.
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)
        row = 0

        grid.addWidget(QLabel(_("Reference photo")), row, 0)
        self._reference_field = QLineEdit()
        self._reference_field.setReadOnly(True)
        self._reference_field.setPlaceholderText(_("The person to become"))
        grid.addWidget(self._reference_field, row, 1)
        button = QPushButton(_("Choose..."))
        button.setMinimumWidth(96)
        button.clicked.connect(self._pick_reference)
        grid.addWidget(button, row, 2)
        row += 1

        grid.addWidget(QLabel(_("Your video")), row, 0)
        self._driving_field = QLineEdit()
        self._driving_field.setReadOnly(True)
        self._driving_field.setPlaceholderText(
            _("Framed like the reference photo"))
        grid.addWidget(self._driving_field, row, 1)
        button = QPushButton(_("Choose..."))
        button.setMinimumWidth(96)
        button.clicked.connect(self._pick_driving)
        grid.addWidget(button, row, 2)
        row += 1

        grid.addWidget(QLabel(_("Size")), row, 0)
        self._size = QComboBox()
        self._size.addItems([label for label, _b in SIZES])
        self._size.currentIndexChanged.connect(self._update_estimate)
        grid.addWidget(self._size, row, 1, 1, 2)
        row += 1

        grid.addWidget(QLabel(_("Quality")), row, 0)
        self._quality = QComboBox()
        self._quality.addItems([label for label, _s in QUALITY])
        self._quality.setCurrentIndex(1)
        self._quality.currentIndexChanged.connect(self._update_estimate)
        grid.addWidget(self._quality, row, 1, 1, 2)
        row += 1

        grid.addWidget(QLabel(_("Smoothness")), row, 0)
        self._smooth = QComboBox()
        self._smooth.addItems([label for label, _g, _o, _s in SMOOTHNESS])
        self._smooth.setToolTip(_(
            "The model costs the same per frame however fast you play them "
            "back, so generating fewer and filling the gaps in afterwards is "
            "close to a straight speedup. It does invent the in-between "
            "frames, so fast motion can smear."))
        self._smooth.currentIndexChanged.connect(self._update_estimate)
        grid.addWidget(self._smooth, row, 1, 1, 2)
        row += 1

        grid.addWidget(QLabel(_("Render at most")), row, 0)
        self._cap = QDoubleSpinBox()
        self._cap.setRange(0.0, 600.0)
        self._cap.setDecimals(1)
        self._cap.setSuffix(_(" seconds"))
        self._cap.setSpecialValueText(_("the whole clip"))
        self._cap.setValue(5.0)
        self._cap.setToolTip(_(
            "Cost is close to linear in length. Set to 0 for the whole clip."))
        self._cap.valueChanged.connect(self._update_estimate)
        grid.addWidget(self._cap, row, 1, 1, 2)
        row += 1

        grid.addWidget(QLabel(_("Memory")), row, 0)
        self._cache = QComboBox()
        self._cache.addItems([label for label, _d, _t in CACHE])
        self._cache.setToolTip(_(
            "The pose branch can be cached so it runs once per pass instead of "
            "once per step. It is a speedup bought with system RAM, and this "
            "machine has little to spare once the 15.5 GB model is loaded."))
        self._cache.currentIndexChanged.connect(self._update_estimate)
        grid.addWidget(self._cache, row, 1, 1, 2)
        row += 1

        grid.addWidget(QLabel(_("Seed")), row, 0)
        self._seed = QSpinBox()
        self._seed.setRange(-1, 2_000_000_000)
        self._seed.setValue(-1)
        self._seed.setToolTip(_("-1 picks a new one each run"))
        grid.addWidget(self._seed, row, 1, 1, 2)
        column.addLayout(grid)

        self._preview = QLabel()
        self._preview.setObjectName("imageDrop")
        self._preview.setFixedHeight(150)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setText(_("No reference chosen"))
        column.addWidget(self._preview)
        column.addStretch(1)
        return column

    def _build_prompts(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(6)

        label = QLabel(_("Who they are, and where"))
        label.setObjectName("cardTitle")
        column.addWidget(label)
        note = QLabel(_(
            "Looks and setting only — never motion. The background is generated "
            "from this text, so describe the one you want."))
        note.setObjectName("hint")
        note.setWordWrap(True)
        column.addWidget(note)
        self._prompt = QPlainTextEdit(wa.DEFAULT_PROMPT)
        self._prompt.setMinimumHeight(150)
        column.addWidget(self._prompt, 1)

        label = QLabel(_("What they're doing"))
        label.setObjectName("cardTitle")
        column.addWidget(label)
        note = QLabel(_("Describe the motion in your video."))
        note.setObjectName("hint")
        note.setWordWrap(True)
        column.addWidget(note)
        self._pose_prompt = QPlainTextEdit(wa.DEFAULT_POSE_PROMPT)
        self._pose_prompt.setFixedHeight(64)
        column.addWidget(self._pose_prompt)
        return column

    # ── inputs ───────────────────────────────────────────────────────────

    def _pick_reference(self) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("Choose a reference photo"),
            RECENT_DIRS["source"] or "", IMAGE_FILTER)
        if not path:
            return
        self._reference = path
        self._reference_field.setText(os.path.basename(path))
        RECENT_DIRS["source"] = os.path.dirname(path)
        try:
            self._preview.setPixmap(render_image_preview(path, (150, 150)))
            self._preview.setText("")
        except Exception:
            self._preview.setText(os.path.basename(path))
        self._check_ready()

    def _pick_driving(self) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("Choose a video of you moving"),
            RECENT_DIRS["target"] or "", VIDEO_FILTER)
        if not path:
            return
        self._driving = path
        self._driving_field.setText(os.path.basename(path))
        RECENT_DIRS["target"] = os.path.dirname(path)
        try:
            self._clip = wa.probe_video(path)
        except Exception:
            self._clip = None
        self._update_estimate()
        self._check_ready()

    # ── settings ─────────────────────────────────────────────────────────

    def _settings(self) -> wa.TransferSettings:
        budget = SIZES[self._size.currentIndex()][1]
        if self._clip:
            _frames, _fps, width, height = self._clip
        else:
            width, height = 480, 848
        out_w, out_h = wa.fit_dimensions(width, height, budget)
        device, dtype = CACHE[self._cache.currentIndex()][1:]
        _label, gen_fps, out_fps, _saving = SMOOTHNESS[
            self._smooth.currentIndex()]
        seed = self._seed.value()
        return wa.TransferSettings(
            width=out_w, height=out_h,
            fps=gen_fps, output_fps=out_fps,
            steps=QUALITY[self._quality.currentIndex()][1],
            seed=None if seed < 0 else seed,
            prompt=self._prompt.toPlainText().strip() or wa.DEFAULT_PROMPT,
            pose_prompt=(self._pose_prompt.toPlainText().strip()
                         or wa.DEFAULT_POSE_PROMPT),
            cache_device=device, cache_dtype=dtype,
            max_seconds=self._cap.value()).snapped()

    def _driving_frames(self, settings: wa.TransferSettings) -> int:
        """Frames after resampling to the chosen generation rate."""
        if not self._clip:
            return wa.SEGMENT_FRAMES
        frames, fps, _w, _h = self._clip
        return max(1, int(frames * (settings.fps / max(fps, 1e-3))))

    def _update_estimate(self) -> None:
        settings = self._settings()
        frames = self._driving_frames(settings)
        if settings.max_seconds:
            frames = min(frames, int(settings.max_seconds * settings.fps))
        passes = len(wa.plan_segments(frames))
        seconds = frames / settings.fps

        parts = [_("{w} x {h}, {s:.1f} s of output, {p} generation pass{plural}."
                   ).format(w=settings.width, h=settings.height, s=seconds,
                            p=passes, plural="" if passes == 1 else "es")]
        window = wa.estimate_minutes(settings, self._driving_frames(settings))
        if window is None:
            parts.append(_("Render time is measured from your first finished "
                           "render rather than guessed."))
        else:
            low, high = window
            parts.append(_("Expect roughly {low:.0f}-{high:.0f} minutes."
                           ).format(low=low, high=high))
        warning = self._resource_warning(settings)
        if warning:
            parts.append(warning)
        self._estimate.setText("  ".join(parts))

    def _check_ready(self) -> None:
        ready = bool(self._reference and self._driving)
        if not wa.backend_installed():
            self._status.setText(_(
                "The render backend is not installed. Expected it at {path}."
                ).format(path=wa.BACKEND_ROOT))
            ready = False
        else:
            missing = wa.missing_models()
            if missing:
                self._status.setText(_(
                    "Still missing {n} model file(s): {names}. They total about "
                    "23 GB and download once.").format(
                        n=len(missing), names=", ".join(missing)))
                ready = False
        self._btn_start.setEnabled(ready and self._thread is None)

    # ── run ──────────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        if self._thread is not None:
            return
        settings = self._settings()
        ensure_user_dirs()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output = os.path.join(CAPTURES_DIR, f"motion-{stamp}.mp4")

        self._cancel.clear()
        self._started_at = time.time()
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._bar.setValue(0)

        def work() -> None:
            try:
                result = wa.transfer(
                    self._reference, self._driving, output, settings,
                    progress=lambda m, f: self._signals.progress.emit(m, f),
                    should_cancel=self._cancel.is_set,
                    backend=self._backend)
                if result.complete:
                    wa.record_measurement(
                        settings, result.generated_frames, result.seconds)
                self._signals.finished.emit(result, "")
            except Exception as error:
                _log_failure(error)
                self._signals.finished.emit(None, str(error))

        self._thread = threading.Thread(target=work, name="WanAnimate",
                                        daemon=True)
        self._thread.start()

    @staticmethod
    def _resource_warning(settings: wa.TransferSettings) -> str:
        """Flag the two things that actually sink a render on this machine."""
        notes = []
        try:
            import psutil
            available = psutil.virtual_memory().available / 1024 ** 3
            need = wa.CHECKPOINT_GB + wa.CACHE_RAM_GB.get(
                settings.cache_dtype, 0.0)
            if available < need:
                # Weights are memory-mapped, so a shortfall degrades into disk
                # paging rather than failing. Worth saying, not worth blocking.
                notes.append(_(
                    "{gb:.0f} GB of memory is free and this wants about "
                    "{need:.0f} GB — it will run, but page from disk and be "
                    "slower.").format(gb=available, need=need))
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                if free / 1024 ** 3 < 8:
                    notes.append(_(
                        "Only {gb:.1f} GB of video memory is free — a game or "
                        "another app is holding it.").format(
                            gb=free / 1024 ** 3))
        except Exception:
            pass
        return "  ".join(notes)

    def _on_progress(self, message: str, fraction: float) -> None:
        self._bar.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        elapsed = time.time() - self._started_at
        suffix = ""
        if fraction > 0.05:
            remaining = elapsed / fraction - elapsed
            suffix = _("  -  about {m:.0f} min left").format(m=remaining / 60)
        self._status.setText(message + suffix)

    def _on_finished(self, result, error: str) -> None:
        self._thread = None
        self._btn_stop.setEnabled(False)
        self._check_ready()
        self._update_estimate()
        if result is None:
            self._status.setText(
                _("{error}  (details in {log})").format(
                    error=error, log=os.path.basename(LOG_PATH)))
            self._bar.setValue(0)
            return

        self._last_output = result.path
        self._bar.setValue(1000)
        name = os.path.basename(result.path)
        if result.complete:
            self._status.setText(
                _("Saved {name} — {frames} frames in {m:.1f} minutes "
                  "({p:.1f} min per pass).").format(
                      name=name, frames=result.frames,
                      m=result.seconds / 60,
                      p=result.minutes_per_segment))
            update_status(f"Motion transfer finished: {name}")
        else:
            # Partial output beats nothing: a stopped pass does not invalidate
            # the ones that already cost you twenty minutes each.
            self._status.setText(
                _("{reason} Saved the {done} of {total} pass(es) that finished "
                  "as {name} ({frames} frames).").format(
                      reason=result.error, done=result.completed_segments,
                      total=result.segments, name=name, frames=result.frames))
            update_status(f"Motion transfer stopped early: {name}")

    def _on_stop(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._cancel.set()
            self._status.setText(_(
                "Stopping after this step. Finished passes will still be "
                "saved."))
            self._btn_stop.setEnabled(False)

    def _open_output(self) -> None:
        ensure_user_dirs()
        from modules.ui import _open_folder
        _open_folder(CAPTURES_DIR)

    def shutdown(self) -> None:
        """Called when the app closes."""
        self._cancel.set()


def _log_failure(error: BaseException) -> None:
    """Keep the traceback. A failure here costs tens of minutes to reproduce."""
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            handle.write("".join(traceback.format_exception(
                type(error), error, error.__traceback__)))
    except OSError:
        pass
