"""The live loop must notice a source image that changed in place.

Regression test for the random-face bug: that button used to rewrite one
temp file, and the loop compared paths only, so every face after the first
was ignored while the UI showed the new thumbnail.
"""

import os
import time

from modules.live_engine import source_token


def test_none_for_missing_path():
    assert source_token(None) is None
    assert source_token("") is None


def test_stable_for_unchanged_file(tmp_path):
    path = tmp_path / "face.jpg"
    path.write_bytes(b"a" * 128)
    assert source_token(str(path)) == source_token(str(path))


def test_changes_when_contents_are_replaced(tmp_path):
    path = tmp_path / "face.jpg"
    path.write_bytes(b"a" * 128)
    before = source_token(str(path))

    # Same path, different contents — the case that used to be missed.
    time.sleep(0.01)
    path.write_bytes(b"b" * 256)
    after = source_token(str(path))

    assert before != after


def test_changes_when_path_changes(tmp_path):
    first = tmp_path / "one.jpg"
    second = tmp_path / "two.jpg"
    first.write_bytes(b"a" * 128)
    second.write_bytes(b"a" * 128)
    assert source_token(str(first)) != source_token(str(second))


def test_survives_a_deleted_file(tmp_path):
    path = tmp_path / "gone.jpg"
    path.write_bytes(b"a")
    os.remove(path)
    # No exception, and still comparable.
    assert source_token(str(path)) == (str(path), None, None)
