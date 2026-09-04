"""Startup must survive a windowed build's missing stdout.

Regression test for a crash that only appeared in the frozen app: PyInstaller
sets sys.stdout to None for a GUI program, and face_swapper.pre_check() then
drove tqdm straight into `'NoneType' object has no attribute 'write'` before
the window ever opened.
"""

import os
import subprocess
import sys

import pytest

from modules import NullStream, ensure_streams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code: str) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter rooted at the project.

    These two checks go through a subprocess rather than importing in-process
    because the thing being tested *is* startup. face_swapper imports
    modules.core, which imports the UI, which imports utilities — a cycle
    that only resolves in the order the real entry point happens to use.
    Importing it first from a test would exercise an order the app never has.
    """
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                          capture_output=True, timeout=600)


def test_pre_check_does_not_download():
    """pre_check verifies; it must never fetch half a gigabyte inline."""
    result = _run(
        "import modules.core;"
        "from modules.processors.frame import face_swapper;"
        "calls = [];"
        "setattr(face_swapper, 'conditional_download',"
        " lambda *a, **k: calls.append(a));"
        "ok = face_swapper.pre_check();"
        "assert ok is True, 'pre_check failed';"
        "assert calls == [], 'pre_check downloaded at startup';"
        "import os; os._exit(0)"
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")[-800:]


def test_pre_check_survives_a_none_stdout():
    """The exact crash: no stdout, model missing, tqdm in the download path."""
    result = _run(
        "import sys; sys.stdout = None; sys.stderr = None;"
        "import modules.core;"
        "from modules.processors.frame import face_swapper;"
        "ok = face_swapper.pre_check();"
        "import os; os._exit(0 if ok else 3)"
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")[-800:]


def test_null_stream_satisfies_tqdm():
    """The replacement stream has to be good enough for real libraries."""
    null = NullStream()
    tqdm = pytest.importorskip("tqdm")
    with tqdm.tqdm(total=10, file=null) as bar:
        bar.update(5)

    assert null.write("x") == 0
    assert null.isatty() is False
    with pytest.raises(OSError):
        null.fileno()


def test_ensure_streams_replaces_missing_ones(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    ensure_streams()
    assert sys.stdout is not None and sys.stderr is not None
    sys.stdout.write("ok")
    sys.stderr.write("ok")


def test_ensure_streams_leaves_real_ones_alone(monkeypatch):
    sentinel = NullStream()
    monkeypatch.setattr(sys, "stdout", sentinel)
    ensure_streams()
    assert sys.stdout is sentinel


def test_importing_modules_repairs_streams():
    """A fresh interpreter with no streams must still import the package."""
    code = (
        "import sys; sys.stdout = None; sys.stderr = None;"
        "import modules;"
        "assert sys.stdout is not None and sys.stderr is not None;"
        "sys.stdout.write('x');"
        "import os; os._exit(0)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, timeout=300)
    assert result.returncode == 0, result.stderr.decode(errors="replace")[-800:]
