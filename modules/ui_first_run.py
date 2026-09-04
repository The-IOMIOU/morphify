"""First-launch model download.

The installer ships without the ~1.2 GB of ONNX models, so a fresh install
has to fetch them once. Doing that behind a progress dialog is much better
than the app silently failing to swap anything, which is what happens if the
models are simply absent.

The download runs on a worker thread; the dialog only reports progress and
offers to cancel.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QDialog,
    QMessageBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from modules.model_store import (
    MODELS_DIR,
    Model,
    download,
    ensure_user_dirs,
    missing_models,
    space_shortfall,
)
import modules.metadata
from modules.ui_common import _


class _Signals(QObject):
    progress = Signal(str, int, int)   # filename, bytes done, bytes total
    finished = Signal(bool, str)       # success, message


class ModelDownloadDialog(QDialog):
    """Modal progress dialog that fetches any missing models."""

    def __init__(self, models: List[Model], parent=None):
        super().__init__(parent)
        self._models = models
        self._signals = _Signals()
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.setWindowTitle(_("Downloading models"))
        self.setModal(True)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        total_mb = sum(m.approx_bytes for m in models) / 1_048_576
        heading = QLabel(
            _("{app} needs to download {count} model file(s), about "
              "{size:.0f} MB. This happens once.").format(
                  app=modules.metadata.name, count=len(models), size=total_mb)
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)

        self._file_label = QLabel("")
        self._file_label.setObjectName("hint")
        layout.addWidget(self._file_label)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(True)
        layout.addWidget(self._bar)

        self._overall = QLabel("")
        self._overall.setObjectName("hint")
        layout.addWidget(self._overall)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self._btn_cancel = QPushButton(_("Cancel"))
        self._btn_cancel.clicked.connect(self._on_cancel)
        buttons.addWidget(self._btn_cancel)
        layout.addLayout(buttons)

        self._signals.progress.connect(self._on_progress)
        self._signals.finished.connect(self._on_finished)
        self._index = 0

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="ModelDownload", daemon=True)
        self._thread.start()

    # ── worker ───────────────────────────────────────────────────────────

    def _run(self) -> None:
        ensure_user_dirs()
        failed: List[str] = []
        for index, model in enumerate(self._models):
            if self._cancel.is_set():
                self._signals.finished.emit(False, _("Download cancelled."))
                return
            self._index = index

            def report(name: str, done: int, total: int) -> None:
                self._signals.progress.emit(name, done, total)

            if not download(model, progress=report):
                failed.append(model.filename)

        if self._cancel.is_set():
            self._signals.finished.emit(False, _("Download cancelled."))
        elif failed:
            self._signals.finished.emit(
                False,
                _("Could not download: {names}").format(names=", ".join(failed)))
        else:
            self._signals.finished.emit(True, _("All models ready."))

    # ── slots ────────────────────────────────────────────────────────────

    def _on_progress(self, name: str, done: int, total: int) -> None:
        self._file_label.setText(name)
        if total > 0:
            self._bar.setValue(int(1000 * done / total))
            self._bar.setFormat(
                f"{done / 1_048_576:.0f} / {total / 1_048_576:.0f} MB  (%p%)")
        else:
            self._bar.setFormat(f"{done / 1_048_576:.0f} MB")
        self._overall.setText(
            _("File {n} of {total}").format(
                n=self._index + 1, total=len(self._models)))

    def _on_finished(self, success: bool, message: str) -> None:
        self._file_label.setText(message)
        self._btn_cancel.setText(_("Close"))
        if success:
            self._bar.setValue(1000)
            self.accept()
        # On failure the dialog stays open so the message can be read.

    def _on_cancel(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._cancel.set()
            self._btn_cancel.setEnabled(False)
            self._file_label.setText(_("Finishing the current file..."))
        else:
            self.reject()

    def closeEvent(self, event) -> None:
        self._cancel.set()
        event.accept()


def run_if_needed(parent=None) -> bool:
    """Fetch missing required models, showing a dialog. True if all present."""
    pending = missing_models(required_only=True)
    if not pending:
        return True

    # Refuse before starting rather than filling the drive with a partial
    # download and failing on the last few megabytes.
    shortfall = space_shortfall(required_only=True)
    if shortfall > 0:
        QMessageBox.warning(
            parent, _("Not enough space"),
            _("The face swap models need about {need:.1f} GB more room on:"
              "\n\n{path}\n\nOpen Setup > Models > \"Change folder...\" to "
              "put them on another drive.").format(
                  need=shortfall / 1024 ** 3, path=MODELS_DIR),
        )
        return False

    dialog = ModelDownloadDialog(pending, parent)
    dialog.start()
    dialog.exec()
    return not missing_models(required_only=True)
