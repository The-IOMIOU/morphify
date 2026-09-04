"""Full Takeover: wear the whole person, not just their face.

The plain face swap replaces the face and nothing else, so you keep your own
hair, your own skin tone on the neck and shoulders, and your own room. That
reads as a face pasted on you. This mode closes the other three gaps:

* **Hair** — segmented from the source portrait and warped onto your head
  every frame with the same alignment the swap already computes.
* **Skin tone** — their colour statistics transferred onto your visible skin,
  so the neck and shoulders match the new face instead of contradicting it.
* **Their background** — you are matted out of your room and composited onto
  the portrait's own background, with the original occupant painted out.

Deliberately *not* a generative model. Everything here is segmentation plus
2D warping, which is why it runs at camera speed on a mid-range GPU. The
honest limit is the hair: it is a warped 2D cutout, so it holds up for the
roughly frontal framing a webcam gives you and degrades on large head turns.
See ``HAIR_NOTE`` for what is done to soften that.

Models (both fetched into models/takeover/):

* ``faceparser.onnx``  BiSeNet, 19-class CelebAMask-HQ labels. Runs once per
  source portrait, so its cost does not touch the frame budget.
* ``modnet.onnx``      Portrait matting for the live frame. ~16 ms at
  512x288, which is the per-frame price of background replacement.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from modules import imread_unicode
from modules.paths import MODELS_DIR

MODELS_SUBDIR = "takeover"
FACE_PARSER = "faceparser.onnx"
MATTING = "modnet.onnx"

# CelebAMask-HQ label ids, verified against this model's output rather than
# assumed: class 1 sits centre-face, 17 sits above it, 14 at the base.
BACKGROUND = 0
SKIN = 1
BROWS = (2, 3)
EYES = (4, 5)
GLASSES = 6
EARS = (7, 8)
EARRING = 9
NOSE = 10
MOUTH = (11, 12, 13)
NECK = 14
NECKLACE = 15
CLOTH = 16
HAIR = 17
HAT = 18

FACE_CLASSES = (SKIN, *BROWS, *EYES, GLASSES, *EARS, NOSE, *MOUTH)
PERSON_CLASSES = tuple(range(1, 19))

PARSE_SIZE = 512

# MODNet's encoder downsamples by 32; anything else fails outright rather
# than degrading, so input is always snapped to a multiple of this.
MATTE_STRIDE = 32
# 384 rather than 512: a silhouette is a smooth, low-frequency signal, and
# the extra detail cost 8ms a frame for no visible difference once the alpha
# is upscaled and feathered.
MATTE_WIDTH = 384

HAIR_NOTE = (
    "Hair is a warped 2D cutout, not a 3D model. It tracks position, scale "
    "and roll exactly, and is sheared with head yaw to fake some volume, but "
    "a large turn will show its flatness."
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_SESSIONS: dict = {}
_SESSION_LOCK = threading.Lock()


def models_dir() -> str:
    return os.path.join(MODELS_DIR, MODELS_SUBDIR)


def model_path(name: str) -> str:
    return os.path.join(models_dir(), name)


def models_available() -> bool:
    return all(os.path.isfile(model_path(n)) for n in (FACE_PARSER, MATTING))


def _session(name: str):
    """Create (once) an onnxruntime session for one of our models."""
    with _SESSION_LOCK:
        if name in _SESSIONS:
            return _SESSIONS[name]
        import onnxruntime as ort

        from modules.processors.frame._onnx_enhancer import build_provider_config

        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = ort.InferenceSession(
            model_path(name), sess_options=options,
            providers=build_provider_config())
        _SESSIONS[name] = session
        return session


def unload() -> None:
    with _SESSION_LOCK:
        _SESSIONS.clear()


# ─── segmentation ────────────────────────────────────────────────────────


def parse_face(image: np.ndarray) -> np.ndarray:
    """19-class label map for ``image`` (BGR), at the image's own size."""
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(image, (PARSE_SIZE, PARSE_SIZE)),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = ((rgb - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)[None]
    logits = _session(FACE_PARSER).run(None, {"input": blob})[0][0]
    labels = logits.argmax(0).astype(np.uint8)
    return cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)


def matte_person(image: np.ndarray, width: int = MATTE_WIDTH) -> np.ndarray:
    """Soft alpha for the person in ``image`` (BGR), float32 0..1, image size.

    Runs at a reduced width and is upscaled: matting is a smooth signal, so
    the detail lost is not worth three times the frame cost.
    """
    source_height, source_width = image.shape[:2]
    scale = width / max(1, source_width)
    height = max(MATTE_STRIDE, int(round(source_height * scale)))
    # Both axes must land on the stride or the model refuses the input.
    width = int(round(width / MATTE_STRIDE)) * MATTE_STRIDE
    height = int(round(height / MATTE_STRIDE)) * MATTE_STRIDE

    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None]
    alpha = _session(MATTING).run(None, {"input": blob})[0][0, 0]
    alpha = np.clip(alpha, 0.0, 1.0)
    return cv2.resize(alpha, (source_width, source_height),
                      interpolation=cv2.INTER_LINEAR)


def mask_from_labels(labels: np.ndarray, classes) -> np.ndarray:
    mask = np.zeros(labels.shape, dtype=bool)
    for value in classes:
        mask |= labels == value
    return mask


# ─── colour ──────────────────────────────────────────────────────────────


def lab_statistics(image: np.ndarray, mask: np.ndarray,
                   stride: int = 2) -> Optional[tuple]:
    """Mean and standard deviation in LAB over the masked pixels.

    Sampled every ``stride`` pixels: these are summary statistics over tens
    of thousands of pixels, and reading a quarter of them moves the result
    by far less than the lighting difference the transfer is correcting.
    """
    if mask[::stride, ::stride].sum() < 64:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = lab[::stride, ::stride][mask[::stride, ::stride]].astype(np.float32)
    if pixels.size < 64:
        return None
    return pixels.mean(axis=0), pixels.std(axis=0) + 1e-6


def transfer_tone(image: np.ndarray, region: np.ndarray, target_stats: tuple,
                  strength: float = 1.0) -> np.ndarray:
    """Shift ``region`` of ``image`` toward ``target_stats`` in LAB.

    Reinhard-style transfer, applied only where ``region`` is non-zero and
    blended by ``strength`` so it can be dialled back when the lighting in
    the two photographs is very different.

    The correction is affine per LAB channel, which makes it a lookup table:
    building three 256-entry LUTs and running ``cv2.LUT`` replaces a
    full-frame float32 rescale and is several times faster.
    """
    if strength <= 0 or target_stats is None:
        return image
    binary = region > 0.05
    source_stats = lab_statistics(image, binary)
    if source_stats is None:
        return image

    source_mean, source_std = source_stats
    target_mean, target_std = target_stats

    ramp = np.arange(256, dtype=np.float32)
    table = np.empty((256, 1, 3), dtype=np.uint8)
    for channel in range(3):
        gain = float(target_std[channel] / source_std[channel])
        mapped = (ramp - source_mean[channel]) * gain + target_mean[channel]
        table[:, 0, channel] = np.clip(mapped, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    shifted = cv2.cvtColor(cv2.LUT(lab, table), cv2.COLOR_LAB2BGR)

    weight = np.clip(region, 0.0, 1.0) * float(strength)
    return blend(image, shifted, weight)


def blend(base: np.ndarray, overlay: np.ndarray,
          alpha: np.ndarray) -> np.ndarray:
    """Per-pixel alpha blend using cv2's SIMD paths rather than numpy.

    ``base*(1-a) + overlay*a`` in float32 numpy allocates several full-frame
    temporaries; cv2 does the same arithmetic multithreaded and in place.
    """
    weight = alpha if alpha.ndim == 3 else cv2.merge([alpha, alpha, alpha])
    weight = weight.astype(np.float32, copy=False)
    base_f = base.astype(np.float32, copy=False)
    overlay_f = overlay.astype(np.float32, copy=False)
    out = cv2.multiply(cv2.subtract(overlay_f, base_f), weight)
    out = cv2.add(base_f, out)
    return np.clip(out, 0, 255).astype(np.uint8)


# ─── source preparation ──────────────────────────────────────────────────


@dataclass
class SourceAppearance:
    """Everything extracted from the portrait, computed once."""

    path: str
    image: np.ndarray
    labels: np.ndarray
    keypoints: np.ndarray                      # 5x2, the source face
    hair_rgb: np.ndarray                       # full-size BGR
    hair_alpha: np.ndarray                     # full-size float 0..1
    background: np.ndarray                     # portrait with the person removed
    skin_stats: Optional[tuple] = None
    note: str = ""
    size: Tuple[int, int] = field(default=(0, 0))
    # Background scaled to the live frame, kept because the frame size does
    # not change between frames and rescaling a 1024px plate every time is
    # pure waste.
    _plate_cache: dict = field(default_factory=dict, repr=False)

    def plate_for(self, width: int, height: int) -> np.ndarray:
        key = (width, height)
        plate = self._plate_cache.get(key)
        if plate is None:
            plate = _cover(self.background, width, height)
            self._plate_cache.clear()
            self._plate_cache[key] = plate
        return plate


def _feather(mask: np.ndarray, radius: int) -> np.ndarray:
    """Soft-edged float mask from a boolean one."""
    soft = (mask.astype(np.float32) * 255.0)
    radius = max(1, radius | 1)
    soft = cv2.GaussianBlur(soft, (radius, radius), 0)
    return np.clip(soft / 255.0, 0.0, 1.0)


def build_background_plate(image: np.ndarray, person: np.ndarray) -> np.ndarray:
    """The portrait with its occupant painted out.

    Inpainting at full resolution is slow and, for a plate that will sit
    behind a person anyway, wasted. Filling a downscaled copy and enlarging
    it gives a soft, plausible backdrop for a fraction of the time — and
    this runs once, when the source is chosen.
    """
    height, width = image.shape[:2]
    scale = 512 / max(height, width)
    if scale < 1.0:
        small = cv2.resize(image, (int(width * scale), int(height * scale)),
                           interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(person, (small.shape[1], small.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
    else:
        small, small_mask = image.copy(), person

    # Grow the hole so the subject's own edge pixels do not bleed back in.
    kernel = np.ones((9, 9), np.uint8)
    hole = cv2.dilate((small_mask > 0.3).astype(np.uint8) * 255, kernel, iterations=2)
    filled = cv2.inpaint(small, hole, 7, cv2.INPAINT_TELEA)
    filled = cv2.medianBlur(filled, 5)

    if scale < 1.0:
        filled = cv2.resize(filled, (width, height), interpolation=cv2.INTER_LINEAR)
    return filled


def prepare_source(path: str) -> Optional[SourceAppearance]:
    """Extract hair, skin tone and a clean background from a portrait."""
    image = imread_unicode(path)
    if image is None:
        return None

    from modules.face_analyser import get_source_face

    face = get_source_face(image)
    if face is None or getattr(face, "kps", None) is None:
        return None

    # get_source_face translates padded-detection coordinates back, so
    # these keypoints index this image.
    keypoints = np.asarray(face.kps, dtype=np.float32)

    labels = parse_face(image)
    hair_mask = mask_from_labels(labels, (HAIR, HAT))
    person_mask = mask_from_labels(labels, PERSON_CLASSES)

    # The parser only knows the head and shoulders; matting finds the rest of
    # the body, so the plate does not keep a floating torso.
    try:
        person_alpha = np.maximum(
            person_mask.astype(np.float32), matte_person(image))
    except Exception:
        person_alpha = person_mask.astype(np.float32)

    note = ""
    if hair_mask.sum() < image.shape[0] * image.shape[1] * 0.005:
        note = "Barely any hair found in this portrait — a hat or a tight " \
               "crop will do that."

    hair_alpha = _feather(hair_mask, radius=max(3, image.shape[0] // 120))
    skin_stats = lab_statistics(image, mask_from_labels(labels, (SKIN,)))

    return SourceAppearance(
        path=path,
        image=image,
        labels=labels,
        keypoints=keypoints,
        hair_rgb=image,
        hair_alpha=hair_alpha,
        background=build_background_plate(image, person_alpha),
        skin_stats=skin_stats,
        note=note,
        size=(image.shape[1], image.shape[0]),
    )


# ─── per-frame compositing ───────────────────────────────────────────────


def _yaw_from_keypoints(keypoints: np.ndarray) -> float:
    """Rough head yaw in -1..1 from the nose's offset between the eyes.

    Not a pose model — just enough signal to lean the hair the right way.
    """
    left_eye, right_eye, nose = keypoints[0], keypoints[1], keypoints[2]
    centre = (left_eye + right_eye) / 2.0
    span = np.linalg.norm(right_eye - left_eye) + 1e-6
    return float(np.clip((nose[0] - centre[0]) / span * 2.0, -1.0, 1.0))


def warp_hair(source: SourceAppearance, keypoints: np.ndarray,
              frame_shape: Tuple[int, int],
              volume: float = 1.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Map the source's hair onto a head at ``keypoints``.

    A similarity transform from the five source keypoints to the five target
    ones carries position, scale and roll. Yaw is not in that transform — the
    keypoints flatten as the head turns — so the hair is additionally sheared
    and widened by the yaw difference. That is the "2.5D" part: it makes the
    silhouette swing the right way instead of staying rigidly frontal.
    """
    if keypoints is None or len(keypoints) < 5:
        return None, None

    matrix, _ = cv2.estimateAffinePartial2D(
        source.keypoints, np.asarray(keypoints, dtype=np.float32),
        method=cv2.LMEDS)
    if matrix is None:
        return None, None

    if volume > 0:
        yaw_delta = (_yaw_from_keypoints(np.asarray(keypoints, np.float32))
                     - _yaw_from_keypoints(source.keypoints))
        # Pivot around the eye midpoint so the shear rotates the hair mass
        # rather than sliding it off the head.
        pivot = np.asarray(keypoints, np.float32)[:2].mean(axis=0)
        shear = np.array([
            [1.0, yaw_delta * 0.18 * volume, 0.0],
            [0.0, 1.0 + abs(yaw_delta) * 0.04 * volume, 0.0],
        ], dtype=np.float32)
        shear[0, 2] = pivot[0] - (shear[0, 0] * pivot[0] + shear[0, 1] * pivot[1])
        shear[1, 2] = pivot[1] - (shear[1, 0] * pivot[0] + shear[1, 1] * pivot[1])

        full = np.vstack([matrix, [0, 0, 1]]).astype(np.float32)
        full = np.vstack([shear, [0, 0, 1]]).astype(np.float32) @ full
        matrix = full[:2]

    height, width = frame_shape[:2]
    hair = cv2.warpAffine(source.hair_rgb, matrix, (width, height),
                          flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    alpha = cv2.warpAffine(source.hair_alpha, matrix, (width, height),
                           flags=cv2.INTER_LINEAR, borderValue=0)
    return hair, np.clip(alpha, 0.0, 1.0)


def composite(base: np.ndarray, overlay: np.ndarray,
              alpha: np.ndarray) -> np.ndarray:
    """Alpha-blend ``overlay`` onto ``base``."""
    return blend(base, overlay, alpha)


def _cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale ``image`` to fill (width, height), cropping the overflow."""
    source_height, source_width = image.shape[:2]
    scale = max(width / source_width, height / source_height)
    resized = cv2.resize(
        image, (max(width, int(np.ceil(source_width * scale))),
                max(height, int(np.ceil(source_height * scale)))),
        interpolation=cv2.INTER_LINEAR)
    top = (resized.shape[0] - height) // 2
    left = (resized.shape[1] - width) // 2
    return resized[top:top + height, left:left + width]


def _alpha_bounds(alpha: np.ndarray, pad: int = 12) -> Optional[tuple]:
    """Bounding box of the non-transparent area, padded, or None if empty."""
    rows = np.any(alpha > 0.02, axis=1)
    cols = np.any(alpha > 0.02, axis=0)
    if not rows.any() or not cols.any():
        return None
    y1, y2 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    x1, x2 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    height, width = alpha.shape[:2]
    return (max(0, int(x1) - pad), max(0, int(y1) - pad),
            min(width, int(x2) + pad), min(height, int(y2) + pad))


def apply(frame: np.ndarray, source: SourceAppearance, keypoints,
          hair: bool = True, skin: bool = True, background: bool = True,
          skin_strength: float = 0.8, hair_volume: float = 1.0,
          face_box=None, matte=None) -> np.ndarray:
    """Put the source's hair, tone and backdrop onto an already-swapped frame.

    Called after the face swap, so ``frame`` already carries the new face.
    Each stage is independent and skippable — they are separate switches in
    the UI because they fail in different ways and on different footage.

    Tone transfer and compositing are confined to the person's bounding box.
    A person typically covers a third of the frame, and doing full-frame
    arithmetic for them was the single largest cost in this function.
    """
    height, width = frame.shape[:2]
    result = frame
    hair_alpha = None

    if hair:
        hair_rgb, hair_alpha = warp_hair(
            source, keypoints, frame.shape, volume=hair_volume)
        if hair_rgb is not None:
            box = _alpha_bounds(hair_alpha)
            if box is not None:
                x1, y1, x2, y2 = box
                result = frame.copy()
                result[y1:y2, x1:x2] = blend(
                    frame[y1:y2, x1:x2], hair_rgb[y1:y2, x1:x2],
                    hair_alpha[y1:y2, x1:x2])

    if not (skin or background):
        return result

    try:
        person_alpha = (matte or matte_person)(result)
    except Exception:
        # Without a matte there is no silhouette to cut or tone to correct;
        # the hair still stands on its own.
        return result

    if hair_alpha is not None:
        # The transplanted hair is part of the person now, or compositing
        # would slice it off at the original silhouette.
        person_alpha = np.maximum(person_alpha, hair_alpha)

    bounds = _alpha_bounds(person_alpha)
    if bounds is None:
        return result
    bx1, by1, bx2, by2 = bounds

    if skin and source.skin_stats is not None:
        skin_region = person_alpha.copy()
        if face_box is not None:
            # The swapped face already carries their colour; correcting it
            # again would push it past the source.
            x1, y1, x2, y2 = [int(v) for v in face_box]
            pad_x = int((x2 - x1) * 0.12)
            pad_y = int((y2 - y1) * 0.12)
            x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x); y2 = min(height, y2 + pad_y)
            skin_region[y1:y2, x1:x2] = 0.0
        if hair_alpha is not None:
            skin_region = np.clip(skin_region - hair_alpha, 0.0, 1.0)
        patch = transfer_tone(
            result[by1:by2, bx1:bx2], skin_region[by1:by2, bx1:bx2],
            source.skin_stats, strength=skin_strength)
        if result is frame:
            result = frame.copy()
        result[by1:by2, bx1:bx2] = patch

    if background:
        plate = source.plate_for(width, height)
        composed = plate.copy()
        composed[by1:by2, bx1:bx2] = blend(
            plate[by1:by2, bx1:bx2], result[by1:by2, bx1:bx2],
            person_alpha[by1:by2, bx1:bx2])
        result = composed

    return result


class Compositor:
    """Per-session state for the live takeover path.

    Exists so the matte can be reused across frames. A person's silhouette
    changes far more slowly than their expression, so re-matting every frame
    spends 13 ms to recompute something almost identical — the same reasoning
    behind re-detecting faces only every few frames.
    """

    def __init__(self, matte_every: int = 2) -> None:
        self.matte_every = max(1, matte_every)
        self._counter = 0
        self._alpha: Optional[np.ndarray] = None
        self._alpha_shape: Optional[tuple] = None

    def reset(self) -> None:
        self._alpha = None
        self._alpha_shape = None
        self._counter = 0

    def matte(self, frame: np.ndarray) -> np.ndarray:
        shape = frame.shape[:2]
        stale = (self._alpha is None
                 or self._alpha_shape != shape
                 or self._counter % self.matte_every == 0)
        self._counter += 1
        if stale:
            self._alpha = matte_person(frame)
            self._alpha_shape = shape
        return self._alpha

    def process(self, frame: np.ndarray, source: SourceAppearance,
                keypoints, **options) -> np.ndarray:
        return apply(frame, source, keypoints, matte=self.matte, **options)


# ─── source cache ────────────────────────────────────────────────────────


class SourceCache:
    """Prepares appearances off the live thread.

    ``prepare_source`` takes seconds — parsing, matting and inpainting the
    portrait — which is fine once but must never happen inside the frame
    loop. Callers ask for the current appearance every frame and get None
    until it is ready, so the swap keeps running while it builds.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Optional[SourceAppearance] = None
        self._pending: Optional[str] = None
        self._failed: dict = {}
        self._thread: Optional[threading.Thread] = None

    def get(self, path: Optional[str]) -> Optional[SourceAppearance]:
        """Appearance for ``path``, kicking off a build if needed."""
        if not path:
            return None
        with self._lock:
            if self._current is not None and self._current.path == path:
                return self._current
            if self._failed.get(path):
                return None
            if self._pending == path:
                return None
            self._pending = path
            self._thread = threading.Thread(
                target=self._build, args=(path,),
                name="TakeoverPrep", daemon=True)
            self._thread.start()
            return None

    def _build(self, path: str) -> None:
        appearance = None
        try:
            appearance = prepare_source(path)
        except Exception as exc:
            print(f"[takeover] could not prepare {os.path.basename(path)}: {exc}")
        with self._lock:
            if appearance is not None:
                self._current = appearance
            else:
                self._failed[path] = True
            if self._pending == path:
                self._pending = None

    @property
    def is_building(self) -> bool:
        with self._lock:
            return self._pending is not None

    def invalidate(self) -> None:
        with self._lock:
            self._current = None
            self._failed.clear()


cache = SourceCache()
