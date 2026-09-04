"""Make the GPU runtime libraries loadable before onnxruntime needs them.

onnxruntime-gpu dlopens its CUDA provider, which links against cuDNN and
cuBLAS. Those ship as pip wheels under ``site-packages/nvidia/*/bin`` rather
than being installed system-wide, and since Python 3.8 Windows ignores PATH
when resolving an extension module's native dependencies — ``os.add_dll_
directory()`` is the only thing that works.

This has to happen before the first ``InferenceSession`` is constructed. It
used to live in ``run.py``, which meant CUDA silently fell back to CPU for
any entry point that was not that script (tests, tooling, a frozen build with
its own bootstrap). Living in a module that ``modules/__init__.py`` imports
makes it unconditional: importing anything under ``modules`` is enough.

``register()`` is idempotent and never raises — a machine without the wheels
simply gets CPU execution, which is the correct fallback.
"""

from __future__ import annotations

import glob
import os
import platform
import sys
from typing import List

_registered = False

# Populated by register() so callers can report what was actually wired up.
registered_dirs: List[str] = []


def _site_package_roots() -> List[str]:
    """Candidate site-packages dirs, including a project-local venv."""
    roots = []
    if platform.system() == "Windows":
        roots.append(os.path.join(sys.prefix, "Lib", "site-packages"))
    else:
        py_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
        roots.append(os.path.join(sys.prefix, "lib", py_lib, "site-packages"))

    # A venv sitting next to the project, used when the interpreter running
    # us is not that venv.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if platform.system() == "Windows":
        roots.append(os.path.join(project_root, "venv", "Lib", "site-packages"))
    else:
        py_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
        roots.append(os.path.join(project_root, "venv", "lib", py_lib, "site-packages"))

    # PyInstaller lays the wheel contents out beside the executable instead.
    if getattr(sys, "frozen", False):
        roots.append(os.path.join(os.path.dirname(sys.executable), "_internal"))
        roots.append(getattr(sys, "_MEIPASS", ""))

    return [root for root in roots if root and os.path.isdir(root)]


def _candidate_dirs(site_packages: str) -> List[str]:
    """Directories inside a site-packages tree holding GPU shared libraries."""
    found = []
    sub = "bin" if platform.system() == "Windows" else "lib"

    torch_lib = os.path.join(site_packages, "torch", "lib")
    if os.path.isdir(torch_lib):
        found.append(torch_lib)

    nvidia_dir = os.path.join(site_packages, "nvidia")
    if os.path.isdir(nvidia_dir):
        for package in sorted(os.listdir(nvidia_dir)):
            lib_dir = os.path.join(nvidia_dir, package, sub)
            if os.path.isdir(lib_dir):
                found.append(lib_dir)
    return found


def register() -> List[str]:
    """Wire up the GPU library directories. Safe to call more than once."""
    global _registered
    if _registered:
        return registered_dirs
    _registered = True

    seen = set()
    for site_packages in _site_package_roots():
        for lib_dir in _candidate_dirs(site_packages):
            if lib_dir in seen:
                continue
            seen.add(lib_dir)

            if platform.system() == "Windows":
                # PATH covers child processes (ffmpeg); add_dll_directory is
                # what the loader actually consults for extension modules.
                os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(lib_dir)
                except (OSError, AttributeError):
                    continue
            else:
                # LD_LIBRARY_PATH cannot be changed after the process starts,
                # so load the objects directly into the global namespace.
                existing = os.environ.get("LD_LIBRARY_PATH", "")
                if lib_dir not in existing.split(os.pathsep):
                    os.environ["LD_LIBRARY_PATH"] = (
                        lib_dir + (os.pathsep + existing if existing else "")
                    )
                import ctypes
                for so in sorted(glob.glob(os.path.join(lib_dir, "lib*.so*"))):
                    try:
                        ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass

            registered_dirs.append(lib_dir)

    if platform.system() == "Windows":
        _register_openvino()

    return registered_dirs


def _register_openvino() -> None:
    """Let onnxruntime's OpenVINO provider find openvino.dll, if installed.

    Every failure here is non-fatal: OpenVINO simply is not present, and
    onnxruntime falls back to another provider.
    """
    try:
        from onnxruntime.tools.add_openvino_win_libs import (  # type: ignore[import-untyped]  # noqa: E501
            add_openvino_libs_to_path,
        )
        add_openvino_libs_to_path()
    except (ImportError, FileNotFoundError):
        pass
    except SystemExit as exc:
        # add_openvino_libs_to_path() calls sys.exit() when it cannot locate
        # the libraries. Report it, but keep startup alive.
        print(f"[startup] OpenVINO DLL registration skipped: {exc}", flush=True)
