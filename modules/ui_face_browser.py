"""Find and collect source faces without leaving the app.

Three ways in:

* **Search** the web through key-free, documented image APIs (Wikimedia
  Commons, Openverse). Results save under the search term, so the library
  stays searchable — a folder of ``lebron-james-01.jpg`` is findable, one of
  ``random-20260902-153001.jpg`` is not.
* **Generate** synthetic portraits from thispersondoesnotexist.com. Nobody
  depicted exists, which sidesteps the consent question entirely.
* **From a link** — paste an image URL you already have.

This is deliberately not a general embedded web browser. Shipping a full
Chromium (QtWebEngine) would add hundreds of megabytes and a large attack
surface to solve "fetch an image". Fetching is restricted to direct image
responses, with a size cap.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from urllib.parse import urlparse

import requests
from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from modules.image_search import PROVIDERS, SearchError, SearchResult, search_all, search, slugify
from modules.paths import FACES_DIR, ensure_user_dirs
from modules.ui_common import _, pil_to_qpixmap, update_status

GENERATOR_URL = "https://thispersondoesnotexist.com/random-person.jpeg"

THUMB = 132
COLUMNS = 6

# Refuse anything implausible for a portrait; keeps a bad link from pulling
# down a huge file.
MAX_IMAGE_BYTES = 40 * 1024 * 1024

# Magic bytes for the formats Pillow and OpenCV both handle here.
_SIGNATURES = (
    (b"\xff\xd8\xff", "jpg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"BM", "bmp"),
    (b"RIFF", "webp"),
    (b"GIF8", "gif"),
)


def detect_format(data: bytes) -> Optional[str]:
    for signature, extension in _SIGNATURES:
        if data.startswith(signature):
            if extension == "webp" and data[8:12] != b"WEBP":
                continue
            return extension
    return None


def fetch_image(url: str, timeout: int = 25) -> bytes:
    """Download an image, or raise ValueError explaining what went wrong."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http and https links are supported.")

    response = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout, stream=True)
    response.raise_for_status()

    declared = response.headers.get("Content-Length")
    if declared and int(declared) > MAX_IMAGE_BYTES:
        raise ValueError("That file is too large to be a portrait.")

    chunks, total = [], 0
    for chunk in response.iter_content(65536):
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            raise ValueError("That file is too large to be a portrait.")
    data = b"".join(chunks)

    if detect_format(data) is None:
        raise ValueError(
            "That link did not return an image. Use a direct link to a "
            "picture file, not to a page containing one.")
    return data


class _Signals(QObject):
    # index into the candidate list, thumbnail bytes
    thumbnail = Signal(int, bytes)
    searched = Signal(list, str)      # results, error message
    generated = Signal(bytes)
    finished = Signal(str)
    faceChecked = Signal(int, int)    # index, face count


class _Candidate:
    """One image on offer, with whatever we know about it so far."""

    def __init__(self, image_url: str, thumbnail_url: str = "",
                 label: str = "", stem: str = "face", data: Optional[bytes] = None):
        self.image_url = image_url
        self.thumbnail_url = thumbnail_url or image_url
        self.label = label
        self.stem = stem
        self.data = data           # populated for generated / pasted images


class FaceBrowserDialog(QDialog):
    """Search, generate or paste; click the ones to keep."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signals = _Signals()
        self._candidates: List[_Candidate] = []
        self._tiles: List[QPushButton] = []
        self._selected: set = set()
        self._cancel = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=6)
        self._busy = 0
        self.imported: List[str] = []
        # index -> number of faces found in the thumbnail
        self._face_counts: dict = {}
        self._face_queue: queue.Queue = queue.Queue()

        self.setWindowTitle(_("Find faces"))
        self.setModal(True)
        self.resize(THUMB * COLUMNS + 150, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        intro = QLabel(_(
            "Search for a face, generate synthetic ones, or paste a direct "
            "image link. What you keep is saved under the search term, so you "
            "can find it again."))
        intro.setObjectName("hint")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Search row.
        search_row = QHBoxLayout()
        search_row.setSpacing(8)
        self._query = QLineEdit()
        self._query.setPlaceholderText(_("Search a name, e.g. \"lebron james portrait\""))
        self._query.returnPressed.connect(self._on_search)
        search_row.addWidget(self._query, 1)

        self._provider = QComboBox()
        self._provider.addItem(_("All sources"))
        self._provider.addItems(list(PROVIDERS))
        self._provider.setToolTip(_("Where to search"))
        search_row.addWidget(self._provider)

        self._btn_search = QPushButton(_("Search"))
        self._btn_search.setObjectName("primary")
        self._btn_search.clicked.connect(self._on_search)
        search_row.addWidget(self._btn_search)
        layout.addLayout(search_row)

        # Generate / link row.
        extra_row = QHBoxLayout()
        extra_row.setSpacing(8)
        extra_row.addWidget(QLabel(_("Synthetic")))
        self._count = QSpinBox()
        self._count.setRange(1, 40)
        self._count.setValue(8)
        extra_row.addWidget(self._count)
        self._btn_generate = QPushButton(_("Generate"))
        self._btn_generate.setToolTip(
            _("Portraits of people who do not exist"))
        self._btn_generate.clicked.connect(self._on_generate)
        extra_row.addWidget(self._btn_generate)

        extra_row.addSpacing(14)
        self._url = QLineEdit()
        self._url.setPlaceholderText(_("https://example.com/portrait.jpg"))
        self._url.returnPressed.connect(self._on_fetch_url)
        extra_row.addWidget(self._url, 1)
        btn_url = QPushButton(_("Fetch link"))
        btn_url.clicked.connect(self._on_fetch_url)
        extra_row.addWidget(btn_url)
        layout.addLayout(extra_row)

        # Results grid.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        self._grid = QGridLayout(body)
        self._grid.setSpacing(8)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        self._status = QLabel("")
        self._status.setObjectName("hint")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        buttons = QHBoxLayout()
        self._btn_faces = QPushButton(_("Select faces"))
        self._btn_faces.setToolTip(
            _("Select every result that actually contains a face"))
        self._btn_faces.clicked.connect(self._on_select_faces)
        buttons.addWidget(self._btn_faces)
        self._btn_all = QPushButton(_("Select all"))
        self._btn_all.clicked.connect(self._on_select_all)
        buttons.addWidget(self._btn_all)
        self._btn_clear = QPushButton(_("Clear results"))
        self._btn_clear.clicked.connect(self._clear_results)
        buttons.addWidget(self._btn_clear)
        buttons.addStretch(1)
        btn_close = QPushButton(_("Close"))
        btn_close.clicked.connect(self.reject)
        buttons.addWidget(btn_close)
        self._btn_keep = QPushButton(_("Add to library"))
        self._btn_keep.setObjectName("primary")
        self._btn_keep.clicked.connect(self._on_keep)
        buttons.addWidget(self._btn_keep)
        layout.addLayout(buttons)

        self._signals.thumbnail.connect(self._on_thumbnail)
        self._signals.searched.connect(self._on_searched)
        self._signals.generated.connect(self._on_generated)
        self._signals.finished.connect(self._on_finished)
        self._signals.faceChecked.connect(self._on_face_checked)
        threading.Thread(target=self._face_worker, name="FaceCheck",
                         daemon=True).start()
        self._update_buttons()
        self._query.setFocus()

    # ── searching ────────────────────────────────────────────────────────

    def _on_search(self) -> None:
        query = self._query.text().strip()
        if not query or self._busy:
            return
        self._set_busy(True)
        self._status.setText(_("Searching for \"{q}\"...").format(q=query))
        provider = self._provider.currentText()

        def work() -> None:
            try:
                if provider == _("All sources"):
                    results = search_all(query, limit=36)
                else:
                    results = search(query, provider=provider, limit=36)
                self._signals.searched.emit(results, "")
            except SearchError as exc:
                self._signals.searched.emit([], str(exc))
            finally:
                self._signals.finished.emit("")

        threading.Thread(target=work, name="FaceSearch", daemon=True).start()

    def _on_searched(self, results: List[SearchResult], error: str) -> None:
        if error:
            self._status.setText(error)
            return
        if not results:
            self._status.setText(_("Nothing found. Try a different wording."))
            return

        stem = slugify(self._query.text())
        start = len(self._candidates)
        for result in results:
            self._add_candidate(_Candidate(
                image_url=result.image_url,
                thumbnail_url=result.thumbnail_url,
                label=f"{result.label}  ({result.width}x{result.height})",
                stem=stem,
            ))
        self._status.setText(
            _("{n} results. Click the ones you want, then Add to library."
              ).format(n=len(results)))
        self._queue_thumbnails(start)

    def _queue_thumbnails(self, start: int) -> None:
        for index in range(start, len(self._candidates)):
            candidate = self._candidates[index]
            if candidate.data is not None:
                self._signals.thumbnail.emit(index, candidate.data)
                continue

            def work(i=index, url=candidate.thumbnail_url) -> None:
                if self._cancel.is_set():
                    return
                try:
                    data = fetch_image(url, timeout=20)
                except Exception:
                    return
                if not self._cancel.is_set():
                    self._signals.thumbnail.emit(i, data)

            self._pool.submit(work)

    # ── generating ───────────────────────────────────────────────────────

    def _on_generate(self) -> None:
        if self._busy:
            return
        wanted = self._count.value()
        self._set_busy(True)
        self._status.setText(_("Generating..."))

        def work() -> None:
            problem = ""
            for _index in range(wanted):
                if self._cancel.is_set():
                    break
                try:
                    self._signals.generated.emit(fetch_image(GENERATOR_URL))
                except (requests.RequestException, ValueError) as exc:
                    problem = str(exc)
                    break
                # The endpoint repeats itself if hit too quickly.
                time.sleep(0.6)
            self._signals.finished.emit(problem)

        threading.Thread(target=work, name="FaceGenerator", daemon=True).start()

    def _on_generated(self, data: bytes) -> None:
        index = len(self._candidates)
        self._add_candidate(_Candidate(
            image_url="", label=_("synthetic"), stem="synthetic", data=data))
        self._signals.thumbnail.emit(index, data)

    # ── pasted link ──────────────────────────────────────────────────────

    def _on_fetch_url(self) -> None:
        url = self._url.text().strip()
        if not url:
            return
        self._status.setText(_("Fetching..."))
        try:
            data = fetch_image(url)
        except (requests.RequestException, ValueError) as exc:
            self._status.setText(str(exc))
            return
        stem = slugify(os.path.splitext(os.path.basename(urlparse(url).path))[0]
                       or "linked")
        index = len(self._candidates)
        self._add_candidate(_Candidate(
            image_url=url, label=url, stem=stem, data=data))
        self._signals.thumbnail.emit(index, data)
        self._url.clear()
        self._status.setText(_("Added to the grid below."))

    # ── grid ─────────────────────────────────────────────────────────────

    def _add_candidate(self, candidate: _Candidate) -> None:
        index = len(self._candidates)
        self._candidates.append(candidate)

        button = QPushButton()
        button.setObjectName("faceTile")
        button.setCheckable(True)
        button.setFixedSize(THUMB, THUMB)
        button.setIconSize(QSize(THUMB - 8, THUMB - 8))
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setToolTip(candidate.label)
        button.setText("...")
        button.toggled.connect(lambda on, i=index: self._on_toggled(i, on))
        self._grid.addWidget(button, index // COLUMNS, index % COLUMNS)
        self._tiles.append(button)
        self._update_buttons()

    def _on_thumbnail(self, index: int, data: bytes) -> None:
        if index >= len(self._tiles):
            return
        try:
            from io import BytesIO

            from PIL import Image, ImageOps
            image = Image.open(BytesIO(data))
            image = ImageOps.fit(image, (THUMB - 8, THUMB - 8), Image.LANCZOS)
            self._tiles[index].setIcon(QIcon(pil_to_qpixmap(image)))
            self._tiles[index].setText("")
        except Exception:
            self._tiles[index].setText("?")
            return
        self._face_queue.put((index, data))

    def _face_worker(self) -> None:
        """Count faces in each thumbnail, one at a time.

        Serialised on purpose: this is a *face* finder, so knowing which
        results actually contain a usable portrait is the point — but the
        detector is shared with the live pipeline, so it gets one caller.
        """
        while not self._cancel.is_set():
            try:
                index, data = self._face_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                import cv2
                import numpy as np

                from modules.face_analyser import get_many_faces, get_source_face
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                faces = get_many_faces(image) or []
                count = len(faces)
                if count == 0 and get_source_face(image) is not None:
                    # A tight headshot the plain detector cannot see.
                    count = 1
            except Exception:
                continue
            if not self._cancel.is_set():
                self._signals.faceChecked.emit(index, count)

    def _on_face_checked(self, index: int, count: int) -> None:
        if index >= len(self._tiles):
            return
        self._face_counts[index] = count
        tile = self._tiles[index]
        tile.setProperty("hasface", count > 0)
        tile.style().unpolish(tile)
        tile.style().polish(tile)
        candidate = self._candidates[index]
        if count == 0:
            note = _("no face detected")
        elif count == 1:
            note = _("1 face")
        else:
            note = _("{n} faces").format(n=count)
        tile.setToolTip(f"{candidate.label}\n{note}")
        self._update_buttons()

    def _on_toggled(self, index: int, checked: bool) -> None:
        if checked:
            self._selected.add(index)
        else:
            self._selected.discard(index)
        self._update_buttons()

    def _on_select_faces(self) -> None:
        for index, tile in enumerate(self._tiles):
            tile.setChecked(self._face_counts.get(index, 0) > 0)

    def _on_select_all(self) -> None:
        select = len(self._selected) < len(self._tiles)
        for tile in self._tiles:
            tile.setChecked(select)

    def _clear_results(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._candidates = []
        self._tiles = []
        self._selected = set()
        self._face_counts = {}
        self._status.setText("")
        self._update_buttons()

    def _on_finished(self, problem: str) -> None:
        self._set_busy(False)
        if problem:
            self._status.setText(problem)

    def _set_busy(self, busy: bool) -> None:
        self._busy = 1 if busy else 0
        self._btn_search.setEnabled(not busy)
        self._btn_generate.setEnabled(not busy)

    def _update_buttons(self) -> None:
        count = len(self._selected)
        self._btn_keep.setEnabled(count > 0)
        self._btn_keep.setText(
            _("Add {count} to library").format(count=count) if count
            else _("Add to library"))
        self._btn_all.setEnabled(bool(self._tiles))
        self._btn_clear.setEnabled(bool(self._tiles))
        with_faces = sum(1 for n in self._face_counts.values() if n > 0)
        self._btn_faces.setEnabled(with_faces > 0)
        self._btn_faces.setText(
            _("Select faces ({n})").format(n=with_faces) if with_faces
            else _("Select faces"))

    # ── keep ─────────────────────────────────────────────────────────────

    def _unique_path(self, stem: str, extension: str) -> str:
        for number in range(1, 1000):
            candidate = os.path.join(FACES_DIR, f"{stem}-{number:02d}.{extension}")
            if not os.path.exists(candidate):
                return candidate
        return os.path.join(
            FACES_DIR, f"{stem}-{int(time.time())}.{extension}")

    def _on_keep(self) -> None:
        ensure_user_dirs()
        saved: List[str] = []
        failed = 0
        self._status.setText(_("Downloading..."))

        for index in sorted(self._selected):
            candidate = self._candidates[index]
            data = candidate.data
            if data is None:
                # Only now fetch the full-resolution original.
                try:
                    data = fetch_image(candidate.image_url, timeout=40)
                except Exception:
                    failed += 1
                    continue
            extension = detect_format(data) or "jpg"
            path = self._unique_path(candidate.stem, extension)
            try:
                with open(path, "wb") as handle:
                    handle.write(data)
                saved.append(path)
            except OSError:
                failed += 1

        self.imported = saved
        message = f"Added {len(saved)} face(s) to the library"
        if failed:
            message += f"; {failed} could not be downloaded"
        update_status(message)
        if saved:
            self.accept()
        else:
            self._status.setText(_("Nothing could be downloaded."))

    def closeEvent(self, event) -> None:
        self._cancel.set()
        self._pool.shutdown(wait=False)
        event.accept()
