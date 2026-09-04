"""Secondary windows: the still/video preview and the face mappers.

Moved out of ``ui.py`` during the redesign.  The behaviour is unchanged from
the original implementation — only the imports and the shared-state access
were adjusted so the dialogs no longer reach into the main window's module.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import modules.globals
from modules import imread_unicode
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    add_blank_map,
    get_source_face,
    has_valid_map,
    simplify_maps,
)
from modules.processors.frame.core import get_frame_processors_modules
from modules.ui_common import (
    RECENT_DIRS,
    _,
    bgr_to_qpixmap,
    fit_image_to_size,
    make_thumb,
)
from modules.utilities import is_image, is_video

PREVIEW_DEFAULT_WIDTH = 960
PREVIEW_DEFAULT_HEIGHT = 540
POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
MAPPER_PREVIEW_SIZE = 100

# Set by ui.init() so dialogs can parent themselves to the main window and
# hand control back to it on submit.
_MAIN: Optional[QWidget] = None
_MAPPER: Optional["MapperDialog"] = None
_LIVE_MAPPER: Optional["LiveMapperDialog"] = None

# Set by ui.init(): called with a camera index once a live mapping is
# submitted. Keeps this module from importing the main window.
_START_LIVE: Optional[Callable[[int], None]] = None


def bind_main_window(main: QWidget, start_live: Callable[[int], None]) -> None:
    global _MAIN, _START_LIVE
    _MAIN = main
    _START_LIVE = start_live


# ─── still / video preview ───────────────────────────────────────────────


class PreviewWindow(QWidget):
    """Scrubbable preview of the processed target image or video."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("Preview"))
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._image_label, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.valueChanged.connect(self.refresh_frame)
        self._slider.hide()
        layout.addWidget(self._slider)

    def init_for_target(self) -> None:
        target = modules.globals.target_path
        if is_video(target):
            total = get_video_frame_total(target)
            self._slider.setRange(0, max(0, total - 1))
            self._slider.setValue(0)
            self._slider.show()
        else:
            self._slider.hide()

    def refresh_frame(self, frame_number: int = 0) -> None:
        target = modules.globals.target_path
        source = modules.globals.source_path
        if not target or not source:
            return

        if is_video(target):
            frame = get_video_frame(target, frame_number)
        elif is_image(target):
            frame = imread_unicode(target)
        else:
            return
        if frame is None:
            return

        for processor in get_frame_processors_modules(modules.globals.frame_processors):
            frame = processor.process_frame(
                get_source_face(imread_unicode(source)), frame
            )

        frame = fit_image_to_size(frame, self.width(), self.height())
        self._image_label.setPixmap(bgr_to_qpixmap(frame))


# ─── mapper dialogs ──────────────────────────────────────────────────────


class MapperDialog(QDialog):
    """Source × Target mapper for image / video processing."""

    def __init__(self, start_cb: Callable, mapping: list):
        super().__init__(_MAIN)
        self._start_cb = start_cb
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_WIDTH, POPUP_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_submit = QPushButton(_("Submit"))
        btn_submit.setObjectName("primary")
        btn_submit.clicked.connect(self._on_submit)
        layout.addWidget(btn_submit, alignment=Qt.AlignmentFlag.AlignCenter)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn = QPushButton(_("Select source image"))
            btn.setFixedWidth(200)
            btn.clicked.connect(lambda _c, n=row: self._select_source(n))
            grid.addWidget(btn, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setObjectName("imageDrop")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(make_thumb(item["source"]["cv2"], MAPPER_PREVIEW_SIZE))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setObjectName("imageDrop")
            grid.addWidget(tgt_label, row, 3)
            if "target" in item:
                tgt_label.setPixmap(make_thumb(item["target"]["cv2"], MAPPER_PREVIEW_SIZE))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_source(self, row: int) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            RECENT_DIRS["source"] or "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp)",
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_source_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row]["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_submit(self) -> None:
        if has_valid_map():
            self.accept()
            if _MAIN is not None:
                _MAIN.select_output_and_start()
        else:
            self.set_status("Atleast 1 source with target is required!")


class LiveMapperDialog(QDialog):
    """Source × Target mapper for live webcam mode."""

    def __init__(self, camera_index: int, mapping: list):
        super().__init__(_MAIN)
        self._camera_index = camera_index
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_LIVE_WIDTH, POPUP_LIVE_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        for text, slot, obj in (
            (_("Add"), self._on_add, ""),
            (_("Clear"), self._on_clear, ""),
            (_("Submit"), self._on_submit, "primary"),
        ):
            button = QPushButton(text)
            if obj:
                button.setObjectName(obj)
            button.clicked.connect(slot)
            btn_row.addWidget(button)
        layout.addLayout(btn_row)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn_s = QPushButton(_("Select source image"))
            btn_s.setFixedWidth(200)
            btn_s.clicked.connect(lambda _c, n=row: self._select_face(n, "source"))
            grid.addWidget(btn_s, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setObjectName("imageDrop")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(make_thumb(item["source"]["cv2"], MAPPER_PREVIEW_SIZE))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            btn_t = QPushButton(_("Select target image"))
            btn_t.setFixedWidth(200)
            btn_t.clicked.connect(lambda _c, n=row: self._select_face(n, "target"))
            grid.addWidget(btn_t, row, 3)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setObjectName("imageDrop")
            grid.addWidget(tgt_label, row, 4)
            if "target" in item:
                tgt_label.setPixmap(make_thumb(item["target"]["cv2"], MAPPER_PREVIEW_SIZE))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_face(self, row: int, kind: str) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            RECENT_DIRS["source"] or "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp)",
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_source_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row][kind] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_add(self) -> None:
        add_blank_map()
        self._rebuild()
        self.set_status("Please provide mapping!")

    def _on_clear(self) -> None:
        for item in self._map:
            item.pop("source", None)
            item.pop("target", None)
        self._rebuild()
        self.set_status("All mappings cleared!")

    def _on_submit(self) -> None:
        if has_valid_map():
            simplify_maps()
            self.set_status("Mappings successfully submitted!")
            self.accept()
            if _START_LIVE is not None:
                _START_LIVE(self._camera_index)
        else:
            self.set_status("At least 1 source with target is required!")


# ─── dialog lifecycle ────────────────────────────────────────────────────


def open_mapper_dialog(start_cb: Callable, mapping: list) -> None:
    global _MAPPER
    close_mapper_window()
    _MAPPER = MapperDialog(start_cb, mapping)
    _MAPPER.show()


def open_live_mapper_dialog(camera_index: int, mapping: list) -> None:
    global _LIVE_MAPPER
    close_mapper_window()
    _LIVE_MAPPER = LiveMapperDialog(camera_index, mapping)
    _LIVE_MAPPER.show()


def close_mapper_window() -> None:
    global _MAPPER, _LIVE_MAPPER
    if _MAPPER is not None:
        _MAPPER.close()
        _MAPPER = None
    if _LIVE_MAPPER is not None:
        _LIVE_MAPPER.close()
        _LIVE_MAPPER = None


def mapper_is_open() -> bool:
    return (
        (_MAPPER is not None and _MAPPER.isVisible())
        or (_LIVE_MAPPER is not None and _LIVE_MAPPER.isVisible())
    )


def live_mapper_is_open() -> bool:
    return _LIVE_MAPPER is not None and _LIVE_MAPPER.isVisible()


def raise_live_mapper() -> None:
    if _LIVE_MAPPER is not None:
        _LIVE_MAPPER.raise_()
