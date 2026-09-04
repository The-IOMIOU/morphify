"""Self-check that runs inside the app, including a frozen build.

Reachable as ``Morphify.exe --self-test`` (or ``python run.py
--self-test``). It exercises the things that specifically break in a
PyInstaller bundle and are invisible until a user hits them:

* data files that were not collected,
* modules imported dynamically that the bundler never saw,
* GPU libraries that ended up somewhere gpu_paths does not look.

Anything that needs a camera, a face or the network is out of scope here;
those are covered by the scripts under setup/.
"""

from __future__ import annotations

import os
import sys

import modules.metadata


def run() -> int:
    failures = []
    lines = []

    def say(text: str = "") -> None:
        # A windowed PyInstaller build has no usable stdout, so everything is
        # also collected for the log file written at the end.
        lines.append(text)
        try:
            print(text)
        except Exception:
            pass

    def check(label: str, value, good: bool = True) -> None:
        status = "ok  " if good else "FAIL"
        if not good:
            failures.append(label)
        say(f"  [{status}] {label}: {value}")

    say(f"{modules.metadata.name} self-test  (frozen={getattr(sys, 'frozen', False)})")
    say()

    from modules import gpu_paths
    check("gpu library dirs", len(gpu_paths.registered_dirs),
          len(gpu_paths.registered_dirs) > 0)

    import onnxruntime
    providers = onnxruntime.get_available_providers()
    check("onnxruntime providers",
          ", ".join(p.replace("ExecutionProvider", "") for p in providers))

    from modules.processors.frame.face_swapper import _has_ort_cuda
    cuda = _has_ort_cuda()
    check("cuda session", cuda, cuda)

    from modules.paths import MODELS_DIR, USER_DATA_DIR
    check("user data dir", USER_DATA_DIR)
    check("models dir", MODELS_DIR)

    from modules import model_store
    present = [m.filename for m in model_store.MODELS if model_store.is_present(m)]
    missing = [m.filename for m in model_store.missing_models(required_only=True)]
    check("models present", ", ".join(present) or "none")
    check("required models missing", ", ".join(missing) or "none", not missing)

    from modules.virtual_camera import VirtualCamSink
    available, reason = VirtualCamSink.available()
    check("virtual camera", reason, available)

    from modules.processors.frame.core import load_frame_processor_module
    for name in ("face_swapper", "face_enhancer",
                 "face_enhancer_gpen256", "face_enhancer_gpen512"):
        try:
            load_frame_processor_module(name)
            check(f"processor {name}", "imported")
        except BaseException as exc:  # SystemExit included on purpose
            check(f"processor {name}", repr(exc), False)

    import insightface
    check("insightface", insightface.__version__)

    # Portrait animation is the one path that leaves onnxruntime for torch,
    # and a frozen build is where that combination breaks. Actually convert
    # and run the graph rather than just importing the module.
    try:
        import torch

        check("torch", f"{torch.__version__}, cuda={torch.cuda.is_available()}",
              torch.cuda.is_available())
        from modules import live_portrait

        if not live_portrait.models_available():
            check("portrait animation", "models not downloaded yet")
        else:
            module = live_portrait._load_warping()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            feature = torch.zeros(1, 32, 16, 64, 64, device=device, dtype=dtype)
            points = torch.zeros(1, 21, 3, device=device, dtype=dtype)
            with torch.no_grad():
                output = module(feature, points, points)
            shape = tuple(output.shape)
            check("portrait animation", f"generated {shape}",
                  shape == (1, 3, 512, 512))
    except Exception as exc:
        check("portrait animation", repr(exc)[:160], False)

    try:
        from modules.predicter import predict_image  # noqa: F401
        check("nsfw filter", "importable")
    except Exception as exc:
        check("nsfw filter", repr(exc), False)

    from modules.ui_common import get_available_cameras
    check("cameras", ", ".join(get_available_cameras()[1]))

    # Building the window catches missing Qt plugins and stylesheet errors.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
        import modules.ui as ui
        app = QApplication.instance() or QApplication([])
        app.setStyleSheet(ui.stylesheet())
        ui.set_language("en")
        window = ui.MainWindow(lambda: None, lambda *a, **k: None)
        window._shutting_down = True
        window._ui_timer.stop()
        # Compared against the nav list rather than a literal, so adding a page
        # cannot fail this check for the wrong reason. It caught the Motion
        # page being added -- correctly, but only because the number was stale.
        pages = window._pages.count()
        expected = len(ui.NAV_ITEMS)
        check("main window", f"built, {pages} pages (nav lists {expected})",
              pages == expected)
    except Exception as exc:
        check("main window", repr(exc), False)

    say()
    if failures:
        say(f"RESULT: FAIL ({len(failures)}) -> {', '.join(failures)}")
    else:
        say("RESULT: PASS")

    _write_log(lines)
    return 1 if failures else 0


def _write_log(lines) -> None:
    """Drop the report next to the app's data so it can be sent along."""
    try:
        from modules.paths import USER_DATA_DIR, ensure_user_dirs
        ensure_user_dirs()
        path = os.path.join(USER_DATA_DIR, "self-test.log")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        try:
            print(f"\nreport written to {path}")
        except Exception:
            pass
    except Exception:
        pass
