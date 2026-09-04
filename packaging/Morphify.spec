# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Morphify.

One-folder build. Two things need care beyond the defaults:

* The frame processors are loaded through ``importlib.import_module`` at
  runtime (modules/processors/frame/core.py), so PyInstaller's static
  analysis cannot see them — they are listed as hidden imports.
* The CUDA/cuDNN libraries ship as pip wheels under ``nvidia/<pkg>/bin``.
  That directory layout is preserved in the bundle because
  modules/gpu_paths.py walks exactly that structure at startup to register
  the DLL directories.

Models are intentionally not bundled: they are ~1.2 GB and are fetched on
first launch instead.

    pyinstaller packaging/Morphify.spec --noconfirm
"""

import os
import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
SITE_PACKAGES = os.path.join(ROOT, "venv", "Lib", "site-packages")
if not os.path.isdir(SITE_PACKAGES):
    SITE_PACKAGES = os.path.join(sys.prefix, "Lib", "site-packages")

APP_NAME = "Morphify"


# ── GPU runtime libraries ────────────────────────────────────────────────
# Keep the nvidia/<package>/bin layout intact so gpu_paths.register() finds
# them; a flat copy next to the exe would not be discovered.

def collect_nvidia_binaries():
    collected = []
    nvidia_root = os.path.join(SITE_PACKAGES, "nvidia")
    if not os.path.isdir(nvidia_root):
        print("[spec] WARNING: no nvidia wheels found - the build will be "
              "CPU only.")
        return collected
    for package in sorted(os.listdir(nvidia_root)):
        bin_dir = os.path.join(nvidia_root, package, "bin")
        if not os.path.isdir(bin_dir):
            continue
        for name in os.listdir(bin_dir):
            if name.lower().endswith(".dll"):
                collected.append((
                    os.path.join(bin_dir, name),
                    os.path.join("nvidia", package, "bin"),
                ))
    print(f"[spec] bundling {len(collected)} NVIDIA DLLs")
    return collected


binaries = collect_nvidia_binaries()

# torchvision registers custom operators (nms and friends) from a compiled
# extension. Recent releases renamed it from _C.pyd to _C_stable.pyd, which
# PyInstaller's bundled hook does not know about, so it was silently left
# out — onnx2torch then failed at run time with "operator torchvision::nms
# does not exist". Collect them explicitly rather than trusting the hook.
for _package in ("torchvision", "torch"):
    try:
        binaries += collect_dynamic_libs(_package)
    except Exception as _exc:
        print(f"[spec] could not collect {_package} libraries: {_exc}")

# collect_dynamic_libs only looks for .dll on Windows, and these extensions
# are .pyd, so they need naming outright. Without them torch cannot see
# torchvision's operator registrations at all.
_tv_dir = os.path.join(SITE_PACKAGES, "torchvision")
if os.path.isdir(_tv_dir):
    _found = [n for n in os.listdir(_tv_dir) if n.lower().endswith(".pyd")]
    for _name in _found:
        binaries.append((os.path.join(_tv_dir, _name), "torchvision"))
    print(f"[spec] bundling {len(_found)} torchvision extension(s): "
          f"{', '.join(_found) or 'none found'}")


# ── data files ───────────────────────────────────────────────────────────

datas = [
    (os.path.join(ROOT, "locales"), "locales"),
    (os.path.join(ROOT, "packaging", "Morphify.ico"), "."),
]

# insightface ships model configuration and mesh data alongside its code.
datas += collect_data_files("insightface")
# opennsfw2 resolves a weights path relative to its package directory.
datas += collect_data_files("opennsfw2")
datas += collect_data_files("keras")


# ── hidden imports ───────────────────────────────────────────────────────

hiddenimports = [
    # Loaded by name at runtime, never imported statically.
    "modules.processors.frame.face_swapper",
    "modules.processors.frame.face_enhancer",
    "modules.processors.frame.face_enhancer_gpen256",
    "modules.processors.frame.face_enhancer_gpen512",
    "modules.processors.frame.face_masking",
    # Imported lazily inside functions.
    "modules.predicter",
    "modules.onnx_optimize",
    "modules.virtual_camera",
    "modules.recorder",
    "modules.ui_face_browser",
    "modules.ui_first_run",
    "modules.self_test",
    "modules.takeover",
    "modules.face_identity",
    "modules.image_search",
    "modules.live_portrait",
    "torch",
    "torchvision",
    "onnx2torch",
    # Windows camera enumeration.
    "pygrabber.dshow_graph",
    "cv2_enumerate_cameras",
    "comtypes",
    # Keras picks its backend at import time from KERAS_BACKEND.
    "keras",
    "torch",
]
hiddenimports += collect_submodules("insightface")
hiddenimports += collect_submodules("cv2_enumerate_cameras")
# onnx2torch registers its node converters by import side effect;
# PyInstaller sees none of them statically.
hiddenimports += collect_submodules("onnx2torch")


# ── exclusions ───────────────────────────────────────────────────────────

excludes = [
    # The UI is PySide6; the legacy tk paths are unused.
    "tkinter",
    "customtkinter",
    "modules.tkinter_fix",
    "modules.ui_legacy",
    # Notebook/dev tooling that gets pulled in transitively.
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "tensorflow",
    "jax",
    # PySide6 modules the app never touches; each is tens of MB.
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtMultimedia",
    "PySide6.QtQuick3D",
    "PySide6.QtBluetooth",
]


a = Analysis(
    [os.path.join(ROOT, "run.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(ROOT, "packaging", "Morphify.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
