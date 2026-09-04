"""Visual design tokens and the application stylesheet.

Kept apart from the widget code so the look can be adjusted without reading
through layout logic.  Colours are defined once in ``PALETTE`` and
substituted into the stylesheet, so there are no hard-coded hex values
scattered through the QSS.
"""

from __future__ import annotations

PALETTE = {
    # Surfaces, darkest to lightest.
    "bg": "#0e1116",
    "sidebar": "#0a0d11",
    "surface": "#171b22",
    "surface_hi": "#1e242d",
    "border": "#2a313c",
    "border_hi": "#3a434f",

    # Text.
    "text": "#e8ecf2",
    "text_dim": "#8b95a5",
    "text_faint": "#5d6675",

    # Accent and states. Morphify leans violet rather than the plain blue
    # the upstream project used, so the app reads as its own thing.
    "accent": "#7c5cff",
    "accent_hi": "#9a80ff",
    "accent_lo": "#5f3fe0",
    "accent2": "#22d3ee",
    "live": "#2ecc71",
    "live_dim": "#1d7d46",
    "warn": "#f0a02c",
    "danger": "#e5484d",
    "danger_hi": "#f05a5f",
}

# Sizing constants the widget code shares with the stylesheet.
SIDEBAR_WIDTH = 168
WINDOW_MIN_WIDTH = 1120
WINDOW_MIN_HEIGHT = 720
FACE_THUMB_SIZE = 92


_QSS_TEMPLATE = """
QWidget {{
    color: {text};
    font-family: "Segoe UI Variable", "Segoe UI", "SF Pro Text", Arial, sans-serif;
    font-size: 10pt;
}}
QMainWindow, QDialog {{ background-color: {bg}; }}

/* ── header ─────────────────────────────────────────────────────────── */

QFrame#header {{
    background-color: {sidebar};
    border-bottom: 1px solid {border};
}}
QLabel#wordmark {{
    font-size: 13pt;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: {text};
}}
QLabel#wordmarkAccent {{
    font-size: 13pt;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: {accent};
}}

/* Status pills in the header. Colour is set per-state from code by
   swapping the `state` property, so the rules live here rather than in
   inline stylesheets. */
QLabel[pill="true"] {{
    background-color: {surface_hi};
    border: 1px solid {border};
    border-radius: 11px;
    padding: 3px 12px;
    font-size: 9pt;
    font-weight: 600;
    color: {text_dim};
}}
QLabel[pill="true"][state="live"] {{
    background-color: rgba(46, 204, 113, 0.14);
    border-color: {live_dim};
    color: {live};
}}
QLabel[pill="true"][state="busy"] {{
    background-color: rgba(61, 125, 255, 0.14);
    border-color: {accent_lo};
    color: {accent_hi};
}}
QLabel[pill="true"][state="warn"] {{
    background-color: rgba(240, 160, 44, 0.14);
    border-color: {warn};
    color: {warn};
}}
QLabel[pill="true"][state="error"] {{
    background-color: rgba(229, 72, 77, 0.14);
    border-color: {danger};
    color: {danger_hi};
}}

/* ── sidebar ────────────────────────────────────────────────────────── */

QFrame#sidebar {{
    background-color: {sidebar};
    border-right: 1px solid {border};
}}
QPushButton#navItem {{
    background: transparent;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 0;
    padding: 11px 14px;
    text-align: left;
    font-size: 10pt;
    font-weight: 600;
    color: {text_dim};
}}
QPushButton#navItem:hover {{
    background-color: {surface};
    color: {text};
}}
QPushButton#navItem:checked {{
    background-color: {surface};
    border-left: 3px solid {accent};
    color: {text};
}}

/* ── cards and panels ───────────────────────────────────────────────── */

QFrame#card {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 10px;
}}
QLabel#cardTitle {{
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 1.2px;
    color: {text_faint};
}}
QLabel#pageTitle {{
    font-size: 15pt;
    font-weight: 700;
}}
QLabel#hint {{
    color: {text_dim};
    font-size: 9pt;
}}
QLabel#statusLabel {{
    color: {text_dim};
    font-size: 9pt;
}}
QLabel#linkLabel {{
    color: {accent_hi};
}}

/* The live video surface. Black so letterboxed bars read as intentional. */
QLabel#videoSurface {{
    background-color: #000000;
    border: 1px solid {border};
    border-radius: 10px;
    color: {text_faint};
}}

QLabel#imageDrop {{
    background-color: {surface_hi};
    border: 2px dashed {border_hi};
    border-radius: 10px;
    color: {text_faint};
}}
QLabel#imageDrop[dragActive="true"] {{
    border-color: {accent};
    background-color: rgba(61, 125, 255, 0.10);
}}

/* ── buttons ────────────────────────────────────────────────────────── */

QPushButton {{
    background-color: {surface_hi};
    color: {text};
    border: 1px solid {border_hi};
    border-radius: 7px;
    padding: 8px 16px;
    font-weight: 600;
}}
QPushButton:hover {{ background-color: {border}; }}
QPushButton:pressed {{ background-color: {surface}; }}
QPushButton:disabled {{
    background-color: {surface};
    border-color: {border};
    color: {text_faint};
}}

QPushButton#primary {{
    background-color: {accent};
    border-color: {accent};
    color: #ffffff;
}}
QPushButton#primary:hover {{ background-color: {accent_hi}; border-color: {accent_hi}; }}
QPushButton#primary:pressed {{ background-color: {accent_lo}; }}
QPushButton#primary:disabled {{
    background-color: {surface};
    border-color: {border};
    color: {text_faint};
}}

QPushButton#danger {{
    background-color: {danger};
    border-color: {danger};
    color: #ffffff;
}}
QPushButton#danger:hover {{ background-color: {danger_hi}; border-color: {danger_hi}; }}

/* Toggle-style button used for VCam and Live: reads as "on" when checked. */
QPushButton#toggle:checked {{
    background-color: {live};
    border-color: {live};
    color: #08130c;
}}
QPushButton#toggle:checked:hover {{ background-color: #3fdd82; }}

/* Square icon-only button. The default 16px side padding would clip a
   single glyph at this width, so it is dropped here. */
QPushButton#iconButton {{
    padding: 0;
    font-size: 13pt;
}}

/* Mode chips: a segmented control, so the active one must read as chosen
   rather than merely hovered. */
QPushButton#modeChip {{
    background-color: {surface_hi};
    border: 1px solid {border_hi};
    border-radius: 16px;
    padding: 7px 20px;
    font-weight: 600;
    color: {text_dim};
}}
QPushButton#modeChip:hover {{ border-color: {accent}; color: {text}; }}
QPushButton#modeChip:checked {{
    background-color: {accent};
    border-color: {accent};
    color: #ffffff;
}}

QPushButton#faceTile {{
    background-color: {surface_hi};
    border: 2px solid {border};
    border-radius: 8px;
    padding: 0;
}}
QPushButton#faceTile:hover {{ border-color: {border_hi}; }}
QPushButton#faceTile:checked {{ border-color: {accent}; }}
/* Starred faces get a warm edge so favourites are findable at a glance. */
QPushButton#faceTile[starred="true"] {{ border-color: {warn}; }}
QPushButton#faceTile[starred="true"]:checked {{ border-color: {accent}; }}
/* Search results that actually contain a face; the rest stay dim so the
   usable ones are obvious without reading every tooltip. */
QPushButton#faceTile[hasface="true"] {{ border-color: {live_dim}; }}
QPushButton#faceTile[hasface="false"] {{ border-color: {border}; }}
QPushButton#faceTile[hasface="true"]:checked {{ border-color: {accent}; }}

QLineEdit {{
    background-color: {surface_hi};
    border: 1px solid {border_hi};
    border-radius: 7px;
    padding: 6px 10px;
    selection-background-color: {accent};
}}
QLineEdit:focus {{ border-color: {accent}; }}

QProgressBar {{
    background-color: {surface_hi};
    border: 1px solid {border_hi};
    border-radius: 7px;
    height: 20px;
    text-align: center;
    color: {text};
}}
QProgressBar::chunk {{
    background-color: {accent};
    border-radius: 6px;
}}

QMenu {{
    background-color: {surface_hi};
    border: 1px solid {border_hi};
    border-radius: 8px;
    padding: 6px;
}}
QMenu::item {{ padding: 6px 22px; border-radius: 5px; }}
QMenu::item:selected {{ background-color: {accent}; color: #ffffff; }}
QMenu::separator {{ height: 1px; background: {border}; margin: 5px 8px; }}

/* ── inputs ─────────────────────────────────────────────────────────── */

QComboBox, QSpinBox {{
    background-color: {surface_hi};
    border: 1px solid {border_hi};
    border-radius: 7px;
    padding: 6px 10px;
    min-height: 20px;
}}
QComboBox:hover, QSpinBox:hover {{ border-color: {accent}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {surface_hi};
    border: 1px solid {border_hi};
    border-radius: 6px;
    selection-background-color: {accent};
    outline: none;
}}

QCheckBox {{ spacing: 9px; padding: 3px 0; }}
QCheckBox::indicator {{
    width: 34px; height: 18px;
    border-radius: 9px;
    background-color: {border};
    border: 1px solid {border_hi};
}}
QCheckBox::indicator:hover {{ border-color: {border_hi}; }}
QCheckBox::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
}}
QCheckBox:disabled {{ color: {text_faint}; }}

QSlider::groove:horizontal {{
    height: 5px;
    background: {border};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {text};
    width: 15px; height: 15px;
    margin: -6px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
QSlider::sub-page:horizontal {{
    background: {accent};
    border-radius: 3px;
}}
QSlider::groove:horizontal:disabled {{ background: {surface_hi}; }}
QSlider::sub-page:horizontal:disabled {{ background: {border}; }}
QSlider::handle:horizontal:disabled {{ background: {text_faint}; }}

/* ── misc ───────────────────────────────────────────────────────────── */

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {border_hi};
    border-radius: 5px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {text_faint}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

QToolTip {{
    background-color: {surface_hi};
    color: {text};
    border: 1px solid {border_hi};
    border-radius: 6px;
    padding: 5px 8px;
}}

QFrame#separator {{ background-color: {border}; max-height: 1px; border: none; }}
"""


def stylesheet() -> str:
    """The full application stylesheet with palette values substituted in."""
    return _QSS_TEMPLATE.format(**PALETTE)
