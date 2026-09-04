"""Render each UI page to a PNG without a display.

A quick visual check that does not need a camera or a desktop session:
builds the real MainWindow under Qt's offscreen platform plugin and grabs
each page. Used during development to eyeball layout changes.

    python setup/render_ui.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import modules.globals  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_render"
    )
    os.makedirs(out_dir, exist_ok=True)

    modules.globals.execution_providers = ["CUDAExecutionProvider"]
    modules.globals.execution_threads = 2

    import modules.ui as ui

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(ui.stylesheet())
    ui.set_language("en")

    window = ui.MainWindow(lambda: None, lambda *a, **k: None)
    window.resize(1280, 800)
    window.show()
    app.processEvents()

    # Full Takeover has its own option row; render it as an extra shot.
    window._apply_mode("takeover", announce=False)
    app.processEvents()
    window.grab().save(os.path.join(out_dir, "0b_live_takeover.png"))
    print(f"wrote {os.path.join(out_dir, '0b_live_takeover.png')}")
    window._apply_mode("swap", announce=False)
    app.processEvents()

    pages = ["live", "faces", "studio", "setup", "about"]
    for index, name in enumerate(pages):
        window._pages.setCurrentIndex(index)
        window._nav_group.button(index).setChecked(True)
        app.processEvents()
        path = os.path.join(out_dir, f"{index}_{name}.png")
        window.grab().save(path)
        print(f"wrote {path}")

    # Don't run closeEvent's destroy callback path through Qt teardown.
    window._shutting_down = True
    window._ui_timer.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
