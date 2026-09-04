"""Build the frozen application, and optionally the installer.

    python packaging/build.py            # exe only
    python packaging/build.py --installer  # exe + Setup.exe

Wraps PyInstaller rather than calling it directly because Python 3.10.0's
``dis`` module has a bug that crashes PyInstaller's bytecode scanner (see
_patch_dis below).
"""

from __future__ import annotations

import argparse
import dis
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGING = os.path.join(ROOT, "packaging")
SPEC = os.path.join(PACKAGING, "Morphify.spec")
DIST = os.path.join(ROOT, "dist")
BUILD = os.path.join(ROOT, "build")
ISCC = os.path.join(ROOT, "tools", "innosetup", "ISCC.exe")


def _patch_dis() -> None:
    """Make ``dis`` tolerant of the const-index bug in CPython 3.10.0.

    3.10.0 shipped a defect where an EXTENDED_ARG-prefixed LOAD_CONST can be
    decoded with an index past the end of the constants tuple, raising
    IndexError. PyInstaller walks every module's bytecode looking for
    imports, so a single affected module aborts the whole build. Fixed in
    3.10.1, but the venv here is on 3.10.0.

    Returning a placeholder instead of raising is safe for this use: the
    scanner only reads constants to discover import names, and anything it
    misses is covered by the explicit hiddenimports in the spec.
    """
    if sys.version_info[:3] != (3, 10, 0):
        return

    original = dis._get_const_info

    def tolerant(const_index, const_list):
        try:
            return original(const_index, const_list)
        except IndexError:
            return None, repr(None)

    dis._get_const_info = tolerant
    print("[build] applied the CPython 3.10.0 dis workaround")


def clean() -> None:
    for path in (BUILD, os.path.join(DIST, "Morphify")):
        if os.path.isdir(path):
            print(f"[build] removing {path}")
            shutil.rmtree(path, ignore_errors=True)


def build_exe() -> bool:
    _patch_dis()
    from PyInstaller.__main__ import run as pyinstaller_run

    print("[build] running PyInstaller...")
    try:
        pyinstaller_run([
            SPEC,
            "--noconfirm",
            "--distpath", DIST,
            "--workpath", BUILD,
        ])
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(f"[build] PyInstaller failed with exit code {exc.code}")
            return False

    exe = os.path.join(DIST, "Morphify", "Morphify.exe")
    if not os.path.isfile(exe):
        print(f"[build] expected executable not found: {exe}")
        return False

    size = _dir_size(os.path.join(DIST, "Morphify"))
    print(f"[build] built {exe}")
    print(f"[build] bundle size: {size / 1024 ** 3:.2f} GB")
    return True


def build_installer() -> bool:
    if not os.path.isfile(ISCC):
        print(f"[build] Inno Setup compiler not found at {ISCC}")
        print("[build] install Inno Setup 6 from https://jrsoftware.org/isdl.php")
        print(f"[build] then copy its program folder to {os.path.dirname(ISCC)}")
        return False
    iss = os.path.join(PACKAGING, "Morphify.iss")
    print("[build] running Inno Setup...")
    result = subprocess.run([ISCC, iss], cwd=ROOT)
    if result.returncode != 0:
        print(f"[build] ISCC failed with exit code {result.returncode}")
        return False
    print("[build] installer written to dist/")
    return True


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installer", action="store_true",
                        help="also build the Setup.exe with Inno Setup")
    parser.add_argument("--clean", action="store_true",
                        help="remove previous build output first")
    parser.add_argument("--installer-only", action="store_true",
                        help="skip PyInstaller, just rebuild the installer")
    args = parser.parse_args()

    if args.clean:
        clean()

    if not args.installer_only:
        if not build_exe():
            return 1

    if args.installer or args.installer_only:
        if not build_installer():
            return 1

    print("[build] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
