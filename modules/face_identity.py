"""Build one source identity out of several photos of the same person.

The swap model reduces a source face to a single 512-d ArcFace embedding —
``INSwapper.get()`` reads nothing else off it. That embedding carries the
lighting, angle and expression of whichever photo it came from, so a swap
built from one picture tends to look like *that picture* rather than like
the person.

Averaging embeddings across several photos of the same face cancels most of
that per-photo bias and gives a noticeably steadier identity across head
poses. It costs nothing at run time: the blend happens once, and the live
loop still sees a single face object.

Two things make this safe to do automatically:

* **Outlier rejection.** Photos pulled from a web search are not reliably
  the same person — a "kai cenat" search returns other people too. Averaging
  a stranger in would quietly corrupt the identity, so embeddings that
  disagree with the group are dropped rather than blended.
* **Grouping by name.** The face finder saves results under the search term
  (``kai-cenat-01.jpg``, ``kai-cenat-02.jpg``), so files that belong to one
  person are already named as a set.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules import imread_unicode

# Cosine similarity below this against the group's centre means "probably not
# the same person". ArcFace embeddings for one identity typically sit well
# above 0.5 even across pose and lighting; different people land near 0.
OUTLIER_THRESHOLD = 0.35

# Blending more than this brings diminishing returns and slows library scans.
MAX_IMAGES = 12

_CACHE: Dict[tuple, Any] = {}
_LOCK = threading.Lock()

_GROUP_SUFFIX = re.compile(r"[-_ ]?\d{1,3}$")


def group_key(path: str) -> str:
    """Identity a file belongs to, from its name.

    ``kai-cenat-01.jpg`` and ``kai-cenat-07.jpg`` share the key
    ``kai-cenat``; a lone ``holiday.png`` is its own group.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return _GROUP_SUFFIX.sub("", stem).strip("-_ ").lower() or stem.lower()


def find_group(path: str, folder: Optional[str] = None) -> List[str]:
    """Every image in ``folder`` that shares ``path``'s identity."""
    folder = folder or os.path.dirname(path)
    if not os.path.isdir(folder):
        return [path]
    key = group_key(path)
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif")
    members = [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.lower().endswith(extensions)
        and group_key(os.path.join(folder, name)) == key
    ]
    return members or [path]


def _cache_key(paths: Sequence[str]) -> tuple:
    stamped = []
    for path in paths:
        try:
            stat = os.stat(path)
            stamped.append((path, stat.st_size, stat.st_mtime_ns))
        except OSError:
            stamped.append((path, None, None))
    return tuple(stamped)


def _reject_outliers(embeddings: List[np.ndarray],
                     paths: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    """Drop embeddings that disagree with the group's centre."""
    if len(embeddings) < 3:
        # With one or two photos there is no majority to appeal to, so
        # trusting the user's selection is the least surprising behaviour.
        return embeddings, []

    stack = np.stack(embeddings)
    centre = stack.mean(axis=0)
    norm = np.linalg.norm(centre)
    if norm == 0:
        return embeddings, []
    centre = centre / norm

    similarities = stack @ centre
    keep, dropped = [], []
    for embedding, similarity, path in zip(embeddings, similarities, paths):
        if similarity >= OUTLIER_THRESHOLD:
            keep.append(embedding)
        else:
            dropped.append(path)
    # Never reject everything; if the group is incoherent, keep it as-is and
    # let the result speak for itself.
    return (keep, dropped) if keep else (embeddings, [])


def build_identity(paths: Sequence[str],
                   report: Optional[Any] = None) -> Optional[Any]:
    """Return a face whose embedding is the blend of ``paths``.

    The returned object is a real detected face (so bbox, kps and the rest
    stay valid) with its ``normed_embedding`` replaced by the group average.
    Returns None when no usable face is found in any of the images.
    """
    paths = [p for p in paths if p][:MAX_IMAGES]
    if not paths:
        return None

    key = _cache_key(paths)
    with _LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached

    from modules.face_analyser import get_source_face

    faces, embeddings, used = [], [], []
    for path in paths:
        image = imread_unicode(path)
        if image is None:
            continue
        face = get_source_face(image)
        if face is None:
            continue
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            continue
        faces.append(face)
        embeddings.append(np.asarray(embedding, dtype=np.float32))
        used.append(path)

    if not faces:
        return None

    kept, dropped = _reject_outliers(embeddings, used)
    blended = np.stack(kept).mean(axis=0)
    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm

    # Keep the first face as the carrier so every other attribute is real.
    identity = faces[0]
    try:
        identity.normed_embedding = blended.astype(np.float32)
    except Exception:
        return faces[0]

    if report is not None:
        report({
            "used": len(kept),
            "found": len(faces),
            "dropped": dropped,
            "paths": used,
        })

    with _LOCK:
        _CACHE[key] = identity
        if len(_CACHE) > 32:
            _CACHE.pop(next(iter(_CACHE)))
    return identity


def build_from_path(path: str, blend: bool = True) -> Optional[Any]:
    """Source face for ``path``, blended with its name-group when asked."""
    if not path:
        return None
    if not blend:
        image = imread_unicode(path)
        if image is None:
            return None
        from modules.face_analyser import get_source_face
        return get_source_face(image)
    return build_identity(find_group(path))


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
