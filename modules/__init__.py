import os
import sys


class NullStream:
    """A writable stand-in for a missing stdout/stderr.

    PyInstaller sets both to None for a windowed app. Anything that prints
    then raises AttributeError from somewhere unrelated — tqdm's progress
    bar took the whole app down during startup this way, before the window
    existed. Giving them a real file-like object turns a crash into silence.
    """

    def write(self, _text):
        return 0

    def flush(self):
        pass

    def isatty(self):
        return False

    def writable(self):
        return True

    def fileno(self):
        raise OSError("no file descriptor")


def ensure_streams() -> None:
    """Guarantee sys.stdout and sys.stderr are writable. Idempotent."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            setattr(sys, name, NullStream())


ensure_streams()

# Must run before anything constructs an onnxruntime session, so it goes
# ahead of every other import in the package. See modules/gpu_paths.py.
from modules import gpu_paths as _gpu_paths  # noqa: E402

_gpu_paths.register()

import cv2  # noqa: E402
import numpy as np  # noqa: E402


# Utility function to support unicode characters in file paths for reading.
# OpenCV's cv2.imread() encodes the path with the locale ANSI code page on
# Windows, so it silently returns None for paths containing non-ASCII
# characters (Chinese, Japanese, Cyrillic, accents, ...). Reading the bytes
# through NumPy (which uses Python's unicode-aware file I/O) and decoding them
# in memory sidesteps that limitation. Returns None on failure, matching
# cv2.imread() so it stays a drop-in replacement.
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


# Utility function to support unicode characters in file paths for writing.
# cv2.imwrite() has the same ANSI-path limitation, so we encode the image in
# memory and write the bytes out with NumPy's unicode-aware file I/O. Returns
# True/False like cv2.imwrite() so it stays a drop-in replacement.
def imwrite_unicode(path, img, params=None):
    try:
        root, ext = os.path.splitext(path)
        if not ext:
            ext = ".png"
        result, encoded_img = cv2.imencode(ext, img, params if params is not None else [])
        if not result:
            return False
        encoded_img.tofile(path)
        return True
    except Exception:
        return False
