"""How good a source photo is, judged by the one thing that actually matters.

Identity is read by ArcFace at 112x112 and the swap itself runs at 128x128, so
detail beyond roughly 112 px of face width is thrown away -- enlarging an
already-large photo cannot help. Below that it degrades, and below about 64 px
it falls apart. Measured on this machine against a 512 px reference:

    face width   512   160   112    96    64    48    32
    identity    .998  .984  .979  .970  .911  .868  .662

Restoring a small face does not rescue it. GFPGAN and GPEN both scored *worse*
than plain bicubic at every size tested, averaged over four identities:

    face width      bicubic   GFPGAN   GPEN-512
    96                0.966    0.729      0.938
    64                0.924    0.457      0.865
    48                0.821    0.392      0.707
    32                0.676    0.305      0.439

They invent a plausible, generic face instead of recovering the real one, which
is precisely the wrong trade when the whole point is preserving an identity.
Sharper to the eye, further from the person. So there is no repair to offer --
the only useful response to a small source photo is to say so before it costs
someone an evening of wondering why the likeness is poor.
"""

from __future__ import annotations

from typing import Optional, Tuple

#: At or above this, the pipeline already discards the extra detail.
GOOD_FACE_PX = 112

#: Below this, identity starts measurably slipping.
SMALL_FACE_PX = 96

#: Below this it degrades fast -- 0.911 at 64 px, 0.868 at 48, 0.662 at 32.
POOR_FACE_PX = 64


def face_width(face) -> int:
    """Width in pixels of a detected face's bounding box, 0 if unavailable."""
    bbox = getattr(face, "bbox", None)
    if bbox is None or len(bbox) < 4:
        return 0
    try:
        return max(0, int(round(float(bbox[2]) - float(bbox[0]))))
    except (TypeError, ValueError):
        return 0


def verdict(width: int) -> Tuple[str, Optional[str]]:
    """(level, advice) for a source face of this pixel width.

    Levels are "good", "small" and "poor". Advice is None when there is
    nothing worth saying, so callers can stay quiet on a good photo.
    """
    if width <= 0:
        return "unknown", None
    if width >= SMALL_FACE_PX:
        return "good", None
    if width >= POOR_FACE_PX:
        return "small", (
            f"the face is only {width}px wide, so the likeness will be a "
            "little soft. A closer or larger photo helps; enlarging this one "
            "does not.")
    return "poor", (
        f"the face is only {width}px wide. Identity is largely lost below "
        f"{POOR_FACE_PX}px and no amount of upscaling brings it back — find a "
        "photo where the head is bigger in frame.")


def describe(path_name: str, width: int) -> Optional[str]:
    """A status-bar line for a source photo, or None if it is fine."""
    _level, advice = verdict(width)
    if advice is None:
        return None
    return f"{path_name}: {advice}"
