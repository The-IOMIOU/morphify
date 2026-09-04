"""Offline motion transfer: put a photographed person through your movement.

Give it a reference photo and a video of yourself, and it renders a clip of
that person performing your motion — whole body, photoreal, generated rather
than warped.

This is the honest home for what cannot be done live. Photoreal whole-body
re-posing needs a video diffusion model, which is 20-50 network passes per
frame against the face swapper's one. There is no configuration of this that
runs at camera speed, so it does not pretend to: it writes a file.

How it works
------------

1. Your driving video is run through the body pose detector, one skeleton
   per frame. That skeleton *is* the motion; nothing of your appearance is
   carried forward.
2. The skeletons are drawn as an OpenPose-style control video.
3. Wan 2.1 VACE (1.3B) generates a new video conditioned on the control
   video for motion and the reference photo for identity.

The 1.3B model is chosen over the 14B deliberately: it needs about 8 GB of
VRAM against 32 GB+, which is the difference between running on a mid-range
card and not running at all.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from modules import body_pose
from modules.paths import MODELS_DIR

MODEL_SUBDIR = "wan-vace-1.3b"

# Wan's VAE has a temporal stride of 4, and the pipeline wants 4n+1 frames.
FRAME_QUANTUM = 4

# 480p-class output. Both axes must be multiples of 16 for the VAE.
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 832
SIZE_MULTIPLE = 16

DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 5.0
DEFAULT_FRAMES = 49          # 4*12+1, about 3 seconds at 16fps
OUTPUT_FPS = 16

# Deliberately describes *preservation*, not style. An earlier default asked
# for "cinematic lighting, high quality video" and the model duly invented a
# cinematic scene, discarding the reference's own setting and look. The text
# prompt competes with the reference image, so it should say as little as
# possible beyond "keep them the same".
DEFAULT_PROMPT = (
    "the same person as the reference image, same face, same clothing, "
    "same background, full body visible, moving naturally"
)
DEFAULT_NEGATIVE = (
    "blurry, distorted anatomy, extra limbs, missing limbs, deformed hands, "
    "flickering, low quality, watermark, text"
)

# OpenPose-style limb colours. VACE was trained on pose renders in this
# convention, so the colouring is part of the signal, not decoration.
LIMB_COLOURS = {
    (body_pose.LEFT_SHOULDER, body_pose.RIGHT_SHOULDER): (85, 0, 255),
    (body_pose.LEFT_SHOULDER, body_pose.LEFT_ELBOW): (0, 85, 255),
    (body_pose.LEFT_ELBOW, body_pose.LEFT_WRIST): (0, 170, 255),
    (body_pose.RIGHT_SHOULDER, body_pose.RIGHT_ELBOW): (0, 255, 170),
    (body_pose.RIGHT_ELBOW, body_pose.RIGHT_WRIST): (0, 255, 85),
    (body_pose.LEFT_SHOULDER, body_pose.LEFT_HIP): (170, 0, 255),
    (body_pose.RIGHT_SHOULDER, body_pose.RIGHT_HIP): (255, 0, 170),
    (body_pose.LEFT_HIP, body_pose.RIGHT_HIP): (255, 0, 85),
    (body_pose.LEFT_HIP, body_pose.LEFT_KNEE): (255, 85, 0),
    (body_pose.LEFT_KNEE, body_pose.LEFT_ANKLE): (255, 170, 0),
    (body_pose.RIGHT_HIP, body_pose.RIGHT_KNEE): (170, 255, 0),
    (body_pose.RIGHT_KNEE, body_pose.RIGHT_ANKLE): (85, 255, 0),
    (body_pose.NOSE, body_pose.LEFT_EYE): (255, 0, 255),
    (body_pose.NOSE, body_pose.RIGHT_EYE): (255, 0, 200),
    (body_pose.LEFT_EYE, body_pose.LEFT_EAR): (200, 0, 255),
    (body_pose.RIGHT_EYE, body_pose.RIGHT_EAR): (150, 0, 255),
}

_PIPELINE = None
_LOCK = threading.Lock()


def model_dir() -> str:
    return os.path.join(MODELS_DIR, MODEL_SUBDIR)


def model_available() -> bool:
    return os.path.isfile(os.path.join(model_dir(), "model_index.json"))


REPO_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
DOWNLOAD_GB = 17.7


def download_model(progress: Optional[Callable[[str], None]] = None) -> str:
    """Fetch Wan VACE. About 18 GB, so very much opt-in."""
    from huggingface_hub import snapshot_download

    if progress:
        progress(f"Downloading the video model (~{DOWNLOAD_GB:.0f} GB)...")
    return snapshot_download(
        REPO_ID, local_dir=model_dir(), max_workers=4,
        ignore_patterns=["*.md", ".gitattributes"])


def snap(value: int, multiple: int = SIZE_MULTIPLE) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def snap_frames(count: int) -> int:
    """Nearest valid frame count: the VAE needs 4n+1."""
    count = max(FRAME_QUANTUM + 1, int(count))
    return ((count - 1) // FRAME_QUANTUM) * FRAME_QUANTUM + 1


# ─── control video ───────────────────────────────────────────────────────


def draw_pose(pose, width: int, height: int,
              thickness: int = 4) -> np.ndarray:
    """One OpenPose-style skeleton frame on black."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if pose is None:
        return canvas
    for (a, b), colour in LIMB_COLOURS.items():
        if pose.visible(a) and pose.visible(b):
            cv2.line(canvas,
                     tuple(pose.points[a].astype(int)),
                     tuple(pose.points[b].astype(int)),
                     colour, thickness, cv2.LINE_AA)
    for index in range(17):
        if pose.visible(index):
            cv2.circle(canvas, tuple(pose.points[index].astype(int)),
                       thickness, (255, 255, 255), -1, cv2.LINE_AA)
    return canvas


@dataclass
class ControlVideo:
    frames: List[np.ndarray] = field(default_factory=list)
    detected: int = 0
    total: int = 0

    @property
    def coverage(self) -> float:
        return self.detected / self.total if self.total else 0.0


def build_control_video(video_path: str, width: int, height: int,
                        max_frames: int = DEFAULT_FRAMES,
                        stride: int = 1,
                        progress: Optional[Callable[[int, int], None]] = None
                        ) -> ControlVideo:
    """Turn a driving video into pose frames at the output resolution.

    Frames are letterboxed rather than stretched: the generator reproduces
    whatever body proportions the skeleton implies, so distorting the
    skeleton distorts the person.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open {os.path.basename(video_path)}")

    result = ControlVideo()
    try:
        index = 0
        while len(result.frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride:
                index += 1
                continue
            index += 1

            fitted = _letterbox(frame, width, height)
            pose = body_pose.detect_one(fitted)
            result.total += 1
            if pose is not None:
                result.detected += 1
            result.frames.append(draw_pose(pose, width, height))
            if progress:
                progress(len(result.frames), max_frames)
    finally:
        capture.release()

    if not result.frames:
        raise ValueError("No frames could be read from the driving video.")
    return result


def _letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    new_w = max(1, int(round(source_width * scale)))
    new_h = max(1, int(round(source_height * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - new_h) // 2
    left = (width - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


# ─── generation ──────────────────────────────────────────────────────────


def load_pipeline(progress: Optional[Callable[[str], None]] = None):
    """Load Wan VACE with aggressive offloading so it fits in 12 GB."""
    global _PIPELINE
    with _LOCK:
        if _PIPELINE is not None:
            return _PIPELINE
        if not model_available():
            raise FileNotFoundError(
                f"Wan VACE is not downloaded. Expected {model_dir()}")

        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline

        if progress:
            progress("Loading the video model (this takes a minute)...")

        cuda = torch.cuda.is_available()
        free_gb = 0.0
        if cuda:
            free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3

        # The VAE dominates wall-clock time, not the denoising loop: a
        # 6-step render spent 55 seconds generating and ~17 minutes
        # decoding, because an fp32 VAE with tiling forced onto a partly
        # offloaded device. bf16 without tiling is roughly an order of
        # magnitude faster and, on this VAE, visually indistinguishable at
        # 480p. Tiling is kept only as the low-memory fallback it is meant
        # to be.
        vae_dtype = torch.bfloat16 if cuda else torch.float32
        vae = AutoencoderKLWan.from_pretrained(
            model_dir(), subfolder="vae", torch_dtype=vae_dtype)
        pipeline = WanVACEPipeline.from_pretrained(
            model_dir(), vae=vae, torch_dtype=torch.bfloat16)

        if cuda:
            # The text encoder alone is ~10 GB, so it cannot stay resident
            # next to the transformer on a 12 GB card.
            pipeline.enable_model_cpu_offload()
            if free_gb < 7.0:
                if progress:
                    progress(f"Only {free_gb:.1f} GB of video memory free — "
                             "decoding in tiles, which is much slower.")
                try:
                    pipeline.vae.enable_tiling()
                    pipeline.vae.enable_slicing()
                except Exception:
                    pass
        _PIPELINE = pipeline
        return _PIPELINE


def unload() -> None:
    global _PIPELINE
    with _LOCK:
        _PIPELINE = None
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


@dataclass
class TransferSettings:
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    frames: int = DEFAULT_FRAMES
    steps: int = DEFAULT_STEPS
    guidance: float = DEFAULT_GUIDANCE
    conditioning_scale: float = 1.0
    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE
    seed: Optional[int] = None
    fps: int = OUTPUT_FPS

    def normalised(self) -> "TransferSettings":
        return TransferSettings(
            width=snap(self.width), height=snap(self.height),
            frames=snap_frames(self.frames), steps=max(1, self.steps),
            guidance=self.guidance, conditioning_scale=self.conditioning_scale,
            prompt=self.prompt, negative_prompt=self.negative_prompt,
            seed=self.seed, fps=self.fps)


def transfer(reference_path: str, driving_video: str, output_path: str,
             settings: Optional[TransferSettings] = None,
             progress: Optional[Callable[[str, float], None]] = None,
             should_cancel: Optional[Callable[[], bool]] = None) -> str:
    """Render ``reference_path``'s person performing ``driving_video``.

    Returns the written path. Raises with an explanatory message rather than
    a stack trace for the failures a user can actually act on.
    """
    from PIL import Image

    settings = (settings or TransferSettings()).normalised()

    def report(message: str, fraction: float) -> None:
        if progress:
            progress(message, fraction)

    report("Reading the reference photo...", 0.02)
    reference = cv2.imread(reference_path)
    if reference is None:
        raise ValueError("Could not read the reference photo.")

    report("Tracking your movement...", 0.05)
    control = build_control_video(
        driving_video, settings.width, settings.height,
        max_frames=settings.frames,
        progress=lambda done, total: report(
            f"Tracking movement ({done}/{total} frames)...",
            0.05 + 0.15 * done / max(1, total)))

    if control.coverage < 0.5:
        raise ValueError(
            f"A body was only found in {control.coverage:.0%} of the frames. "
            "Stand further back so your whole body is in shot, and make sure "
            "the room is well lit.")

    frames = snap_frames(len(control.frames))
    control.frames = control.frames[:frames]
    while len(control.frames) < frames:
        control.frames.append(control.frames[-1])

    if should_cancel and should_cancel():
        raise InterruptedError("Cancelled.")

    report("Loading the video model...", 0.22)
    pipeline = load_pipeline(lambda m: report(m, 0.22))

    import torch

    video = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
             for f in control.frames]
    # White mask: generate the whole frame, using the control video only for
    # motion and the reference image only for identity.
    mask = [Image.new("L", (settings.width, settings.height), 255)] * frames
    reference_image = Image.fromarray(
        cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))

    generator = None
    if settings.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(settings.seed)

    total_steps = settings.steps

    def on_step(_pipe, step, _timestep, kwargs):
        if should_cancel and should_cancel():
            raise InterruptedError("Cancelled.")
        report(f"Generating frame data (step {step + 1}/{total_steps})...",
               0.25 + 0.65 * (step + 1) / total_steps)
        return kwargs

    report("Generating...", 0.25)
    output = pipeline(
        prompt=settings.prompt,
        negative_prompt=settings.negative_prompt,
        video=video,
        mask=mask,
        reference_images=[reference_image],
        conditioning_scale=settings.conditioning_scale,
        height=settings.height,
        width=settings.width,
        num_frames=frames,
        num_inference_steps=settings.steps,
        guidance_scale=settings.guidance,
        generator=generator,
        callback_on_step_end=on_step,
    ).frames[0]

    report("Writing the video...", 0.93)
    _write_video(output, output_path, settings.fps)
    report("Done.", 1.0)
    return output_path


def _write_video(frames, path: str, fps: int) -> None:
    """Write PIL/array frames out as mp4."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    first = np.asarray(frames[0])
    if first.dtype != np.uint8:
        first = (np.clip(first, 0, 1) * 255).astype(np.uint8)
    height, width = first.shape[:2]

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    if not writer.isOpened() or not os.path.isfile(path):
        raise ValueError(f"Could not open {os.path.basename(path)} to write.")
    try:
        for frame in frames:
            array = np.asarray(frame)
            if array.dtype != np.uint8:
                array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
            writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


# Measured on an RTX 3060 12 GB, 384x672 / 17 frames, bf16 VAE, no tiling:
#   model load  ~1.0 min   (once per session)
#   denoising    0.15 min per step
#   VAE decode  ~5.5 min
# Decode is the dominant cost and does *not* scale with step count, which an
# earlier estimate got wrong by modelling everything as one product — it
# predicted "0-1 min" for a render that took 7.5.
_LOAD_MINUTES = 1.0
_DENOISE_MINUTES_PER_STEP = 0.152
_DECODE_MINUTES = 5.5
_REFERENCE_PIXELS = 384 * 672
_REFERENCE_FRAMES = 17


def estimate_minutes(settings: TransferSettings,
                     model_loaded: bool = False) -> Tuple[float, float]:
    """Wall-clock range in minutes. Deliberately wide; the GPU is shared."""
    settings = settings.normalised()
    scale = ((settings.width * settings.height) / _REFERENCE_PIXELS
             * settings.frames / _REFERENCE_FRAMES)

    denoise = _DENOISE_MINUTES_PER_STEP * settings.steps * scale
    decode = _DECODE_MINUTES * scale
    load = 0.0 if model_loaded else _LOAD_MINUTES

    total = load + denoise + decode
    # +-35%: resolution, VRAM pressure and anything else on the GPU move it.
    return total * 0.7, total * 1.35


def pipeline_loaded() -> bool:
    return _PIPELINE is not None
