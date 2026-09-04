"""Catalogue of the ONNX models the app needs, and how to fetch them.

Lives under ``modules`` rather than ``setup`` so a frozen build can
download models on first launch — the installer ships without them to
keep its size reasonable.

Downloads land in a ``.part`` file and are renamed only once complete, so
an interrupted run can never leave a truncated half-gigabyte model behind
that later looks present and fails at inference time.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional

from modules.paths import MODELS_DIR, ensure_user_dirs, free_bytes  # noqa: F401

_CHUNK = 1 << 18  # 256 KiB


@dataclass(frozen=True)
class Model:
    filename: str
    url: str
    purpose: str
    required: bool
    # Rough expected size in bytes; used only to sanity-check the response,
    # not to verify content.
    approx_bytes: int = 0


MODELS: List[Model] = [
    Model(
        filename="inswapper_128_fp16.onnx",
        url="https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx",
        purpose="Face swap (half precision — preferred on CUDA GPUs)",
        required=True,
        approx_bytes=264_000_000,
    ),
    Model(
        filename="inswapper_128.onnx",
        url="https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx",
        purpose="Face swap (full precision — fallback for older GPUs and CPU)",
        required=True,
        approx_bytes=554_000_000,
    ),
    # The release tag on these is "Models". The URLs baked into
    # face_enhancer_gpen{256,512}.py pointed at a "GPEN-BFR" tag that 404s;
    # both those files and these entries now use the working tag.
    Model(
        filename="GPEN-BFR-256.onnx",
        url="https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GPEN-BFR-256.onnx",
        purpose="Face enhancer (fast — usable live)",
        required=False,
        approx_bytes=75_715_262,
    ),
    Model(
        filename="GPEN-BFR-512.onnx",
        url="https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GPEN-BFR-512.onnx",
        purpose="Face enhancer (higher quality — costly live)",
        required=False,
        approx_bytes=284_250_449,
    ),
    # Full Takeover. Optional, but the mode is inert without them: one
    # segments hair and skin from the portrait, the other cuts you out of
    # your room every frame.
    Model(
        filename="takeover/faceparser.onnx",
        url="https://huggingface.co/bluefoxcreation/Face_parsing_onnx/resolve/main/faceparser_sim.onnx",
        purpose="Full Takeover (hair and skin segmentation)",
        required=False,
        approx_bytes=52_600_000,
    ),
    Model(
        filename="takeover/modnet.onnx",
        url="https://huggingface.co/onnx-community/modnet-webnn/resolve/main/onnx/model.onnx",
        purpose="Full Takeover (background replacement)",
        required=False,
        approx_bytes=25_900_000,
    ),
    # Portrait animation. Large, and only this mode needs them.
    *[
        Model(
            filename=f"liveportrait/{name}.onnx",
            url=("https://huggingface.co/warmshao/FasterLivePortrait/resolve/"
                 f"main/liveportrait_onnx/{name}.onnx"),
            purpose="Portrait animation",
            required=False,
            approx_bytes=size,
        )
        for name, size in (
            ("appearance_feature_extractor", 3_355_896),
            ("motion_extractor", 112_648_514),
            ("warping_spade-fix", 421_233_100),
            ("stitching", 182_363),
        )
    ],
    Model(
        filename="GFPGANv1.4.onnx",
        url="https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GFPGANv1.4.onnx",
        purpose="Face enhancer (GFPGAN — best for stills)",
        required=False,
        approx_bytes=340_256_690,
    ),
]


def is_present(model: Model, models_dir: str = MODELS_DIR) -> bool:
    path = os.path.join(models_dir, model.filename)
    if not os.path.isfile(path):
        return False
    # A file far below the expected size is a leftover from a failed run.
    if model.approx_bytes and os.path.getsize(path) < model.approx_bytes * 0.5:
        return False
    return True


def missing_models(models_dir: str = MODELS_DIR,
                   required_only: bool = False) -> List[Model]:
    return [
        m for m in MODELS
        if (m.required or not required_only) and not is_present(m, models_dir)
    ]


def download(model: Model, models_dir: str = MODELS_DIR,
             progress: Optional[Callable[[str, int, int], None]] = None) -> bool:
    """Download one model. Returns True if it is present afterwards."""
    dest = os.path.join(models_dir, model.filename)
    # Some entries live in a subfolder (takeover/...), so create the parent
    # rather than assuming everything is flat.
    os.makedirs(os.path.dirname(dest) or models_dir, exist_ok=True)
    part = dest + ".part"

    if is_present(model, models_dir):
        return True

    # Clear a stale partial or an undersized previous attempt.
    for stale in (part, dest):
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except OSError:
                pass

    request = urllib.request.Request(
        model.url, headers={"User-Agent": "Deep-Live-Cam"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            with open(part, "wb") as handle:
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(model.filename, done, total)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass
        print(f"  FAILED {model.filename}: {exc}")
        return False

    if total and done < total:
        try:
            os.remove(part)
        except OSError:
            pass
        print(f"  FAILED {model.filename}: truncated ({done}/{total} bytes)")
        return False

    os.replace(part, dest)
    return True


def ensure_models(models_dir: str = MODELS_DIR, required_only: bool = False,
                  progress: Optional[Callable[[str, int, int], None]] = None) -> bool:
    """Download anything missing. Returns True if all required models exist."""
    pending = missing_models(models_dir, required_only)
    for model in pending:
        download(model, models_dir, progress)
    return not missing_models(models_dir, required_only=True)


def space_shortfall(models_dir: str = MODELS_DIR,
                    required_only: bool = False) -> int:
    """Bytes of extra disk space needed for the pending downloads, else 0.

    Checked before starting rather than after: filling a system drive with a
    half-written 500 MB model is a much worse failure than refusing up front.
    """
    pending = missing_models(models_dir, required_only)
    if not pending:
        return 0
    # Headroom for the ".part" copy of the largest file before the rename.
    needed = sum(m.approx_bytes for m in pending)
    needed += max(m.approx_bytes for m in pending)
    available = free_bytes(models_dir)
    if available < 0:
        return 0
    return max(0, needed - available)
