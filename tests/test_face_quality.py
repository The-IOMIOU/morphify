"""Tests for source-photo quality advice.

The thresholds encode a measurement (see modules/face_quality.py), so these
mostly guard against someone quietly moving them to a number that sounds
sensible but is not the one that was measured.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import face_quality as fq  # noqa: E402


class FakeFace:
    def __init__(self, bbox):
        self.bbox = bbox


# ── measuring ────────────────────────────────────────────────────────────────


def test_face_width_reads_the_bounding_box():
    assert fq.face_width(FakeFace([10.0, 20.0, 130.0, 180.0])) == 120


def test_face_width_survives_a_face_without_a_box():
    assert fq.face_width(FakeFace(None)) == 0
    assert fq.face_width(FakeFace([1.0, 2.0])) == 0
    assert fq.face_width(object()) == 0


def test_face_width_survives_nonsense_in_the_box():
    assert fq.face_width(FakeFace(["a", "b", "c", "d"])) == 0


def test_face_width_never_goes_negative():
    assert fq.face_width(FakeFace([200.0, 0.0, 100.0, 100.0])) == 0


# ── advice ───────────────────────────────────────────────────────────────────


def test_a_big_enough_face_gets_no_advice():
    """Staying quiet matters: a warning on every good photo trains people to
    ignore the warning on the bad one."""
    for width in (96, 112, 200, 1000):
        level, advice = fq.verdict(width)
        assert level == "good"
        assert advice is None


def test_a_smallish_face_is_flagged_softly():
    level, advice = fq.verdict(80)
    assert level == "small"
    assert advice and "80px" in advice


def test_a_tiny_face_is_flagged_firmly():
    level, advice = fq.verdict(40)
    assert level == "poor"
    assert advice and "40px" in advice


def test_advice_never_suggests_upscaling_as_a_fix():
    """Measured: every restorer scored worse than bicubic. Advising an upscale
    would send people toward the thing that makes likeness worse."""
    for width in (32, 48, 64, 80, 95):
        _level, advice = fq.verdict(width)
        assert advice is not None
        assert "upscal" not in advice.lower() or "does not" in advice.lower() \
            or "brings it back" in advice.lower()


def test_unknown_width_says_nothing():
    level, advice = fq.verdict(0)
    assert level == "unknown"
    assert advice is None


@pytest.mark.parametrize("width,expected", [
    (fq.POOR_FACE_PX - 1, "poor"),
    (fq.POOR_FACE_PX, "small"),
    (fq.SMALL_FACE_PX - 1, "small"),
    (fq.SMALL_FACE_PX, "good"),
])
def test_thresholds_are_where_the_measurement_put_them(width, expected):
    assert fq.verdict(width)[0] == expected


def test_thresholds_stay_consistent_with_the_models():
    """ArcFace is 112x112; flagging above that would be advice we cannot act on."""
    assert fq.POOR_FACE_PX < fq.SMALL_FACE_PX <= fq.GOOD_FACE_PX == 112


# ── the status line ──────────────────────────────────────────────────────────


def test_describe_names_the_file_it_is_complaining_about():
    message = fq.describe("tom-holland-01.jpg", 40)
    assert message is not None
    assert message.startswith("tom-holland-01.jpg")


def test_describe_is_silent_for_a_good_photo():
    assert fq.describe("fine.jpg", 300) is None
