"""The main window's navigation order.

Kept in its own module, free of Qt and of the rest of the app, for two reasons:
the sidebar buttons and the page stack are built from the same list so they
cannot drift apart, and the indices other code jumps to can be checked by a test
without importing the whole UI.
"""

from __future__ import annotations

#: (icon, title) in sidebar order. Pages are added to the stack in this order.
NAV_ITEMS = [
    ("◉", "Live"),
    ("▤", "Faces"),
    ("✦", "Studio"),
    ("🎬", "Motion"),
    ("⚙", "Setup"),
    ("ⓘ", "About"),
]


def index_of(title: str) -> int:
    for index, (_icon, name) in enumerate(NAV_ITEMS):
        if name == title:
            return index
    raise KeyError(f"no nav page called {title!r}")


LIVE_PAGE_INDEX = index_of("Live")
FACES_PAGE_INDEX = index_of("Faces")
STUDIO_PAGE_INDEX = index_of("Studio")
MOTION_PAGE_INDEX = index_of("Motion")
SETUP_PAGE_INDEX = index_of("Setup")
ABOUT_PAGE_INDEX = index_of("About")
