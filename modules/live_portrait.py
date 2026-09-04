"""LivePortrait: drive a still portrait with your webcam.

Unlike the face swapper, which paints a new face onto your head, this
*generates* the portrait's head in your pose — head turn, expression, eyes,
mouth and the hair inside the crop all move because the model produces them,
not because pixels were pasted.

Why this needs PyTorch when everything else here is onnxruntime
----------------------------------------------------------------

The warping network samples a 5D feature volume with ``grid_sample`` and
resizes 5D tensors. onnxruntime has no CUDA kernel for either, so both fall
back to the CPU and drag a ~90 MB tensor across the bus each time:

    warping+spade, onnxruntime CUDA .......... 637 ms/frame   (1.5 fps)

Upstream solves this with a custom onnxruntime build plus a TensorRT plugin.
Neither is something to inflict on an installer, so instead the ONNX graph is
converted to a torch module once at load time, where those two operations map
straight onto ``F.grid_sample`` and ``F.interpolate`` — both properly
CUDA-accelerated.

The cheap parts of the pipeline (feature extraction, motion, stitching) stay
on onnxruntime, which is already fast for them.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from modules.paths import MODELS_DIR

MODELS_SUBDIR = "liveportrait"

APPEARANCE = "appearance_feature_extractor.onnx"
MOTION = "motion_extractor.onnx"
STITCHING = "stitching.onnx"
WARPING_ONNX = "warping_spade-fix.onnx"

# The model's own input resolution; both the source crop and the driving
# crop are normalised to this before anything else happens.
INPUT_SIZE = 256
OUTPUT_SIZE = 512

_LOCK = threading.Lock()
_SESSIONS: dict = {}
_WARPING = None
_CONVERTERS_REGISTERED = False


def models_dir() -> str:
    return os.path.join(MODELS_DIR, MODELS_SUBDIR)


def model_path(name: str) -> str:
    return os.path.join(models_dir(), name)


def models_available() -> bool:
    return all(os.path.isfile(model_path(n))
               for n in (APPEARANCE, MOTION, STITCHING, WARPING_ONNX))


# ─── torch conversion ────────────────────────────────────────────────────


def _register_converters() -> None:
    """Teach onnx2torch about GridSample.

    onnx2torch ships no converter for it at any opset, which is the only
    thing standing between this graph and a working torch module. The
    mapping is direct: ONNX GridSample and ``F.grid_sample`` take the same
    inputs, the same grid layout and the same three attributes — only the
    spelling of the interpolation modes differs between opset versions.
    """
    global _CONVERTERS_REGISTERED
    if _CONVERTERS_REGISTERED:
        return

    import torch
    from torch import nn
    from onnx2torch.node_converters.registry import add_converter
    from onnx2torch.utils.common import OnnxToTorchModule, OperationConverterResult
    from onnx2torch.utils.common import onnx_mapping_from_node

    mode_map = {
        "bilinear": "bilinear", "linear": "bilinear",
        "nearest": "nearest",
        "bicubic": "bicubic", "cubic": "bicubic",
    }

    class OnnxGridSample(nn.Module, OnnxToTorchModule):
        def __init__(self, mode: str, padding_mode: str, align_corners: bool):
            super().__init__()
            self.mode = mode
            self.padding_mode = padding_mode
            self.align_corners = align_corners

        def forward(self, source: "torch.Tensor", grid: "torch.Tensor"):
            # 5D input is exactly the case onnxruntime cannot do on GPU and
            # torch can; bicubic is 4D-only in torch, so step down for 5D.
            mode = self.mode
            if source.dim() == 5 and mode == "bicubic":
                mode = "bilinear"
            return torch.nn.functional.grid_sample(
                source, grid, mode=mode, padding_mode=self.padding_mode,
                align_corners=self.align_corners)

    def _build(node, graph):  # noqa: ANN001 - onnx2torch's signature
        attributes = node.attributes
        raw_mode = attributes.get("mode", "bilinear")
        if isinstance(raw_mode, bytes):
            raw_mode = raw_mode.decode()
        raw_padding = attributes.get("padding_mode", "zeros")
        if isinstance(raw_padding, bytes):
            raw_padding = raw_padding.decode()
        return OperationConverterResult(
            torch_module=OnnxGridSample(
                mode=mode_map.get(raw_mode, "bilinear"),
                padding_mode=raw_padding,
                align_corners=bool(attributes.get("align_corners", 0)),
            ),
            onnx_mapping=onnx_mapping_from_node(node=node),
        )

    for version in (16, 20, 22):
        try:
            add_converter(operation_type="GridSample", version=version)(_build)
        except Exception:
            # Already registered by a newer onnx2torch; harmless.
            pass

    _CONVERTERS_REGISTERED = True


def _load_warping():
    """Convert the warping+generator graph into a CUDA torch module."""
    global _WARPING
    if _WARPING is not None:
        return _WARPING

    import onnx
    import torch
    from onnx2torch import convert

    _register_converters()

    proto = onnx.load(model_path(WARPING_ONNX))
    # The published graph calls it GridSample3D, which is not an ONNX
    # operator; renaming makes it the standard node our converter handles.
    for node in proto.graph.node:
        if node.op_type == "GridSample3D":
            node.op_type = "GridSample"
            node.domain = ""

    module = convert(proto).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)

    if torch.cuda.is_available():
        # Half precision roughly halves the cost of this conv-heavy graph on
        # a consumer GPU, and TorchScript removes onnx2torch's per-node
        # Python dispatch. Together: 189ms -> ~83ms a frame.
        torch.backends.cudnn.benchmark = True
        module = module.cuda().half()
        try:
            dummy_feature = torch.zeros(1, 32, 16, 64, 64,
                                        device="cuda", dtype=torch.float16)
            dummy_kp = torch.zeros(1, 21, 3, device="cuda", dtype=torch.float16)
            with torch.no_grad():
                traced = torch.jit.trace(
                    module, (dummy_feature, dummy_kp, dummy_kp),
                    check_trace=False)
                module = torch.jit.freeze(traced.eval())
                # Warm the autotuner so the first real frame is not the slow
                # one the user actually sees.
                for _ in range(3):
                    module(dummy_feature, dummy_kp, dummy_kp)
                torch.cuda.synchronize()
        except Exception as exc:
            # Tracing is an optimisation, not a requirement.
            print(f"[live_portrait] running untraced ({exc})")

    _WARPING = module
    return _WARPING


def _session(name: str):
    with _LOCK:
        if name in _SESSIONS:
            return _SESSIONS[name]
        import onnxruntime as ort

        from modules.processors.frame._onnx_enhancer import build_provider_config

        options = ort.SessionOptions()
        options.log_severity_level = 3
        _SESSIONS[name] = ort.InferenceSession(
            model_path(name), sess_options=options,
            providers=build_provider_config())
        return _SESSIONS[name]


def unload() -> None:
    global _WARPING
    with _LOCK:
        _SESSIONS.clear()
    _WARPING = None


# ─── geometry ────────────────────────────────────────────────────────────


def crop_head(image: np.ndarray, keypoints: np.ndarray,
              size: int = INPUT_SIZE, scale: float = 2.3,
              vertical_shift: float = -0.125) -> Tuple[np.ndarray, np.ndarray]:
    """Square, upright crop around a head, plus the transform used.

    LivePortrait expects a generous crop — the head with hair and some
    shoulder, not the tight face box the swapper aligns to — because it
    regenerates everything inside the frame it is given.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    left_eye, right_eye = keypoints[0], keypoints[1]
    mouth = (keypoints[3] + keypoints[4]) / 2.0

    eye_centre = (left_eye + right_eye) / 2.0
    centre = eye_centre + (mouth - eye_centre) * 0.35
    centre = centre + np.array(
        [0.0, vertical_shift * np.linalg.norm(right_eye - left_eye) * scale],
        dtype=np.float32)

    span = float(np.linalg.norm(right_eye - left_eye)) * scale
    span = max(span, 1e-3)
    angle = float(np.arctan2(right_eye[1] - left_eye[1],
                             right_eye[0] - left_eye[0]))

    factor = size / (span * 2.0)
    cos, sin = np.cos(-angle) * factor, np.sin(-angle) * factor
    matrix = np.array([
        [cos, -sin, size / 2.0 - (cos * centre[0] - sin * centre[1])],
        [sin, cos, size / 2.0 - (sin * centre[0] + cos * centre[1])],
    ], dtype=np.float32)

    crop = cv2.warpAffine(image, matrix, (size, size),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return crop, matrix


def _to_blob(crop: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None]


def _rotation(pitch: np.ndarray, yaw: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """Rotation matrix from the model's binned pose logits.

    The three heads emit 66-way distributions rather than angles; the
    expected value over the bins is the angle in degrees.
    """
    bins = np.arange(66, dtype=np.float32)

    def expectation(logits: np.ndarray) -> float:
        exponent = np.exp(logits - logits.max())
        probability = exponent / exponent.sum()
        return float((probability * bins).sum() * 3.0 - 97.5)

    p = np.deg2rad(expectation(pitch[0]))
    y = np.deg2rad(expectation(yaw[0]))
    r = np.deg2rad(expectation(roll[0]))

    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]],
                  dtype=np.float32)
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]],
                  dtype=np.float32)
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]],
                  dtype=np.float32)
    return (rz @ ry @ rx).T


@dataclass
class Motion:
    keypoints: np.ndarray       # 1x21x3, canonical
    rotation: np.ndarray        # 3x3
    expression: np.ndarray      # 1x21x3
    translation: np.ndarray     # 1x3
    scale: float

    def transformed(self) -> np.ndarray:
        """Keypoints placed by this pose: scale * (kp @ R + exp) + t."""
        moved = self.keypoints @ self.rotation + self.expression
        return moved * self.scale + self.translation[:, None, :]


def extract_motion(crop: np.ndarray) -> Motion:
    outputs = _session(MOTION).run(None, {"img": _to_blob(crop)})
    pitch, yaw, roll, translation, expression, scale, keypoints = outputs
    return Motion(
        keypoints=keypoints.reshape(1, 21, 3),
        rotation=_rotation(pitch, yaw, roll),
        expression=expression.reshape(1, 21, 3),
        translation=translation,
        scale=float(scale[0][0]),
    )


def extract_appearance(crop: np.ndarray) -> np.ndarray:
    return _session(APPEARANCE).run(None, {"img": _to_blob(crop)})[0]


@dataclass
class SourcePortrait:
    """Everything extracted once from the still being animated."""

    path: str
    image: np.ndarray
    crop: np.ndarray
    matrix: np.ndarray                 # image -> crop
    appearance: np.ndarray             # 1x32x16x64x64 feature volume
    motion: Motion
    keypoints: np.ndarray              # 1x21x3, posed


def prepare_source(path: str) -> Optional[SourcePortrait]:
    """Crop, feature-extract and pose a portrait. Runs once per source."""
    from modules import imread_unicode
    from modules.face_analyser import get_source_face

    image = imread_unicode(path)
    if image is None:
        return None
    face = get_source_face(image)
    if face is None or getattr(face, "kps", None) is None:
        return None

    crop, matrix = crop_head(image, face.kps)
    motion = extract_motion(crop)
    return SourcePortrait(
        path=path,
        image=image,
        crop=crop,
        matrix=matrix,
        appearance=extract_appearance(crop),
        motion=motion,
        keypoints=motion.transformed(),
    )


def stitch(source_kp: np.ndarray, driving_kp: np.ndarray) -> np.ndarray:
    """Retarget the driving keypoints so the head sits on the source's body.

    Without this the generated head drifts away from the shoulders it is
    supposed to be attached to.
    """
    try:
        pair = np.concatenate(
            [source_kp.reshape(1, -1), driving_kp.reshape(1, -1)], axis=1
        ).astype(np.float32)
        delta = _session(STITCHING).run(None, {"input": pair})[0]
        result = driving_kp.copy()
        result += delta[:, :63].reshape(1, 21, 3)
        result[:, :, :2] += delta[:, 63:65].reshape(1, 1, 2)
        return result
    except Exception:
        return driving_kp


# ─── the animator ────────────────────────────────────────────────────────


class PortraitAnimator:
    """Drives one source portrait from a live camera.

    Uses LivePortrait's *relative* motion formulation: the first driving
    frame becomes a neutral reference, and every later frame contributes only
    its **change** from that reference. Driving a stranger's face with
    absolute pose would otherwise force the portrait into your head shape and
    expression at rest, which looks like a different person.
    """

    def __init__(self) -> None:
        self._source: Optional[SourcePortrait] = None
        self._reference: Optional[Motion] = None
        self._module = None
        self._appearance_gpu = None
        self._source_kp_gpu = None
        self._paste_mask: Optional[np.ndarray] = None

    # ── setup ────────────────────────────────────────────────────────────

    def set_source(self, source: SourcePortrait) -> None:
        import torch

        self._source = source
        self._reference = None
        module = _load_warping()
        self._module = module

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Must match what _load_warping() produced, or conv will refuse the
        # mismatched input and bias dtypes.
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._appearance_gpu = torch.from_numpy(
            source.appearance).to(device=device, dtype=dtype)
        self._source_kp_gpu = torch.from_numpy(
            source.keypoints.astype(np.float32)).to(device=device, dtype=dtype)

    @property
    def ready(self) -> bool:
        return self._source is not None and self._module is not None

    def reset_reference(self) -> None:
        """Take the next driving frame as the new neutral pose."""
        self._reference = None

    # ── per frame ────────────────────────────────────────────────────────

    def _relative_keypoints(self, driving: Motion) -> np.ndarray:
        source = self._source.motion
        reference = self._reference or driving

        rotation = reference.rotation.T @ driving.rotation @ source.rotation
        expression = source.expression + (driving.expression - reference.expression)
        scale = source.scale * (driving.scale / max(reference.scale, 1e-6))
        translation = source.translation + (driving.translation - reference.translation)
        translation = translation.copy()
        # Depth is not meaningful across identities and only causes drift.
        translation[:, 2] = 0.0

        moved = source.keypoints @ rotation + expression
        return moved * scale + translation[:, None, :]

    def _generate(self, driving_kp: np.ndarray) -> np.ndarray:
        import torch

        device = self._appearance_gpu.device
        dtype = self._appearance_gpu.dtype
        kp = torch.from_numpy(driving_kp.astype(np.float32)).to(device, dtype)
        with torch.no_grad():
            out = self._module(self._appearance_gpu, kp, self._source_kp_gpu)
            # Quantise and reorder on the GPU, then move 0.75 MB of uint8
            # across the bus instead of 3 MB of float32.
            frame = (out[0].clamp(0, 1) * 255).to(torch.uint8)
            frame = frame.permute(1, 2, 0).contiguous()
        return cv2.cvtColor(frame.cpu().numpy(), cv2.COLOR_RGB2BGR)

    def _mask(self, size: int) -> np.ndarray:
        """Feathered oval so the generated crop melts into the frame."""
        if self._paste_mask is not None and self._paste_mask.shape[0] == size:
            return self._paste_mask
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(mask, (size // 2, size // 2),
                    (int(size * 0.42), int(size * 0.46)), 0, 0, 360, 255, -1)
        radius = (size // 12) | 1
        mask = cv2.GaussianBlur(mask, (radius, radius), 0)
        self._paste_mask = mask.astype(np.float32) / 255.0
        return self._paste_mask

    def animate(self, frame: np.ndarray, keypoints) -> Optional[np.ndarray]:
        """Return ``frame`` with the source's head driven by this pose."""
        if not self.ready or keypoints is None:
            return None

        crop, matrix = crop_head(frame, keypoints)
        driving = extract_motion(crop)
        if self._reference is None:
            self._reference = driving

        driving_kp = stitch(self._source.keypoints,
                            self._relative_keypoints(driving))
        generated = self._generate(driving_kp)

        # Back into the frame through the inverse of the driving crop.
        height, width = frame.shape[:2]
        scale = OUTPUT_SIZE / INPUT_SIZE
        scaled = matrix.copy()
        scaled[:2, :] *= scale
        inverse = cv2.invertAffineTransform(scaled)

        # The generated head lands in a small part of the frame, so warp and
        # blend only there rather than doing full-frame arithmetic for a
        # region that is mostly untouched.
        corners = np.array([[0, 0], [OUTPUT_SIZE, 0],
                            [OUTPUT_SIZE, OUTPUT_SIZE], [0, OUTPUT_SIZE]],
                           dtype=np.float32)
        mapped = (inverse[:, :2] @ corners.T).T + inverse[:, 2]
        x1 = max(0, int(np.floor(mapped[:, 0].min())) - 2)
        y1 = max(0, int(np.floor(mapped[:, 1].min())) - 2)
        x2 = min(width, int(np.ceil(mapped[:, 0].max())) + 2)
        y2 = min(height, int(np.ceil(mapped[:, 1].max())) + 2)
        if x2 <= x1 or y2 <= y1:
            return frame

        shifted = inverse.copy()
        shifted[0, 2] -= x1
        shifted[1, 2] -= y1
        patch_size = (x2 - x1, y2 - y1)

        warped = cv2.warpAffine(generated, shifted, patch_size,
                                flags=cv2.INTER_LINEAR)
        alpha = cv2.warpAffine(self._mask(OUTPUT_SIZE), shifted, patch_size,
                               flags=cv2.INTER_LINEAR)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]

        result = frame.copy()
        region = frame[y1:y2, x1:x2].astype(np.float32)
        result[y1:y2, x1:x2] = (region * (1.0 - alpha)
                                + warped.astype(np.float32) * alpha).astype(np.uint8)
        return result


# ─── source cache ────────────────────────────────────────────────────────


class SourceCache:
    """Prepares source portraits off the live thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Optional[SourcePortrait] = None
        self._pending: Optional[str] = None
        self._failed: dict = {}

    def get(self, path: Optional[str]) -> Optional[SourcePortrait]:
        if not path:
            return None
        with self._lock:
            if self._current is not None and self._current.path == path:
                return self._current
            if self._failed.get(path) or self._pending == path:
                return None
            self._pending = path
        threading.Thread(target=self._build, args=(path,),
                         name="PortraitPrep", daemon=True).start()
        return None

    def _build(self, path: str) -> None:
        prepared = None
        try:
            prepared = prepare_source(path)
        except Exception as exc:
            print(f"[live_portrait] {os.path.basename(path)}: {exc}")
        with self._lock:
            if prepared is not None:
                self._current = prepared
            else:
                self._failed[path] = True
            if self._pending == path:
                self._pending = None

    def invalidate(self) -> None:
        with self._lock:
            self._current = None
            self._failed.clear()


cache = SourceCache()
