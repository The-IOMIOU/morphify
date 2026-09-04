"""Motion transfer with Wan-Animate-2.

Reference photo (who) + driving video (what motion) -> that character performing
that motion, generated rather than composited. Unlike the pose-skeleton approach
this replaces, the model reads the driving video's *raw frames*, so it keeps
facial expression and hand detail that a stick figure throws away.

The model is a 14-billion-parameter diffusion transformer. Reimplementing its
inference is not realistic and neither is loading it in Morphify's own process,
so it runs in a private ComfyUI backend that Morphify starts on demand and talks
to over HTTP. That backend is a separate checkout from any ComfyUI the user
already has; the two share only a models folder, so nothing here can disturb an
existing install.

What this adds over the reference ComfyUI workflow is the part that workflow
makes you do by hand. The model generates a fixed block of frames per pass, so
covering a clip of any length means chaining passes: each one takes the previous
block's last frame as a temporal anchor and seeks the driving video forward by
exactly one block. The upstream template tells you to copy-paste a subgraph per
block and rewire it yourself. Here it is computed from the clip.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import cv2
import numpy

# ── where things live ────────────────────────────────────────────────────────

BACKEND_ROOT = os.environ.get("MORPHIFY_WAN_BACKEND", r"E:\morphify-backend")
BACKEND_PORT = int(os.environ.get("MORPHIFY_WAN_PORT", "8199"))
BACKEND_HOST = "127.0.0.1"

UNET_NAME = "wan_animate_2_distill_int8_convrot.safetensors"
CLIP_NAME = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
CLIP_VISION_NAME = "clip_vision_h.safetensors"
VAE_NAME = "Wan2_1_VAE_bf16.safetensors"

MODEL_FILES = {
    "diffusion_models": UNET_NAME,
    "text_encoders": CLIP_NAME,
    "clip_vision": CLIP_VISION_NAME,
    "vae": VAE_NAME,
}

# ── model behaviour ──────────────────────────────────────────────────────────

# The model emits this many frames in one pass. It is not a tunable: the value
# is baked into what the network was trained to produce.
SEGMENT_FRAMES = 81

# Wan was trained around 16 fps. Rendering a 30 fps clip costs nearly twice as
# much for motion the model was never trained to resolve, so drive it at 16 and
# play back at 16 -- same wall-clock duration, close to half the work.
DEFAULT_FPS = 16.0

# The distilled checkpoint is trained to run without classifier-free guidance,
# so cfg is 1.0 and each step costs one model pass instead of two.
CFG = 1.0
SAMPLER = "lcm"
SCHEDULER = "simple"
SHIFT = 5.0

# Cost is per generated frame, not per second of video, so generating at a low
# rate and interpolating up buys close to linear time back. It also invents the
# in-between frames, so it is a trade offered rather than one made silently.
# (label, frames generated per second, playback rate, rough fraction of the work)
SMOOTHNESS_PRESETS = [
    ("16 fps - generate every frame, most accurate motion", 16.0, 0.0, 1.0),
    ("12 fps -> smoothed to 24 - about 25% faster", 12.0, 24.0, 0.75),
    ("8 fps -> smoothed to 24 - about twice as fast", 8.0, 24.0, 0.5),
]

#: ``cache_dtype`` value meaning "leave the cache node out entirely".
CACHE_OFF = "none"

#: Roughly what the pose-branch cache costs in system RAM at 480x848/81 frames.
#: Upstream quotes ~12.5 GB in bf16; int8 halves it and int4 quarters it.
CACHE_RAM_GB = {"default": 12.5, "int8": 6.3, "int4": 3.2, CACHE_OFF: 0.0}

#: The checkpoint itself, which exceeds this machine's VRAM and so streams
#: from system RAM on every step.
CHECKPOINT_GB = 15.5

# A 15.5 GB checkpoint on a 24 GB machine does not fit alongside a desktop, and
# the default settings assume it will. Left alone, the backend pins the staged
# weights -- pinned pages cannot be swapped out, so Windows pages everything
# *else* instead and a single sampler step stops finishing at all: measured at
# over twenty minutes for step one, versus seconds of actual compute.
#
# The fix is to stop pretending the weights live in RAM. Unpinned, uncached, and
# told the disk is fast, the backend streams them from the NVMe drive on demand,
# which is a read this machine can do at gigabytes a second.
BACKEND_MEMORY_FLAGS = [
    "--disable-pinned-memory",
    "--cache-none",
    "--fast-disk",
]

# Upstream's standard Wan negative prompt. It is Chinese because that is what
# the text encoder saw in training; translating it measurably weakens it.
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

DEFAULT_PROMPT = (
    "Character appearance description: the person in the reference image, "
    "same face, same hair, same clothing, photorealistic, full body from head "
    "to toe.\n"
    "Background description: the same setting as the reference image, "
    "natural lighting."
)
DEFAULT_POSE_PROMPT = "a person moving naturally, full body visible"


class BackendError(RuntimeError):
    """The ComfyUI backend is missing, unreachable, or rejected the job."""


# ── planning ─────────────────────────────────────────────────────────────────


def valid_length(frames: int) -> int:
    """Round up to a frame count the latent packing accepts (1 mod 4)."""
    frames = max(5, int(frames))
    return ((frames - 1 + 3) // 4) * 4 + 1


@dataclass
class Segment:
    """One generation pass, and where it reads the driving clip."""

    index: int
    #: first driving frame this pass consumes, in the prepared clip
    read_start: int
    #: frames generated by this pass
    length: int
    #: continuation passes regenerate their anchor frame; drop it
    trim_first: bool

    @property
    def new_frames(self) -> int:
        return self.length - (1 if self.trim_first else 0)

    @property
    def read_end(self) -> int:
        return self.read_start + self.length


def plan_segments(total_frames: int, seg_len: int = SEGMENT_FRAMES) -> list[Segment]:
    """Split a driving clip into the passes needed to cover all of it.

    The first pass produces ``seg_len`` frames. Every later pass takes the
    previous pass's final frame as a temporal anchor, *regenerates* it, and
    throws the copy away -- so it advances by ``seg_len - 1``, not ``seg_len``.
    Assuming otherwise leaves the driving video one frame further ahead of the
    anchor on every pass, which does not raise anything; it just stutters a
    little harder at each seam. Hence the iterative recurrence and the tests.
    """
    total_frames = max(1, int(total_frames))
    segments: list[Segment] = []
    read_start = 0
    covered = 0
    while covered < total_frames:
        remaining = total_frames - read_start
        length = seg_len if remaining >= seg_len else valid_length(remaining)
        segments.append(Segment(index=len(segments), read_start=read_start,
                                length=length, trim_first=bool(segments)))
        covered = read_start + length
        read_start = covered - 1  # the next pass re-reads this pass's last frame
        if len(segments) > 400:  # a clip this long is a mistake, not a request
            break
    return segments


@dataclass
class TransferSettings:
    """Everything the user can turn. Defaults match the reference workflow."""

    width: int = 480
    height: int = 848
    steps: int = 10
    fps: float = DEFAULT_FPS
    seed: Optional[int] = None
    prompt: str = DEFAULT_PROMPT
    pose_prompt: str = DEFAULT_POSE_PROMPT
    pose_strength: float = 1.0
    reference_strength: float = 1.0
    #: cache the pose branch so it runs once per pass instead of once per step.
    #: Roughly halves generation time and costs system RAM, so it is a choice.
    cache_device: str = "cpu"
    cache_dtype: str = "int8"
    #: cap on how much of a long clip to render, in seconds. 0 means all of it.
    max_seconds: float = 0.0
    #: play back at this rate, filling the gaps by interpolation. 0 keeps the
    #: generated rate. The model's cost is per *frame*, so generating at 8 and
    #: smoothing to 24 covers three times the footage for the same work.
    output_fps: float = 0.0

    def snapped(self) -> "TransferSettings":
        """Latent packing needs both dimensions on a multiple of 16."""
        self.width = max(16, (int(self.width) // 16) * 16)
        self.height = max(16, (int(self.height) // 16) * 16)
        return self


def fit_dimensions(src_w: int, src_h: int, budget: int = 480 * 848) -> tuple[int, int]:
    """Pick an output size with the clip's aspect ratio under a pixel budget.

    Cost scales with pixels, so the budget is the honest knob; the aspect ratio
    is taken from the driving video because a mismatch between the reference
    framing and the driving framing is the single most common cause of a bad
    result.
    """
    src_w, src_h = max(1, src_w), max(1, src_h)
    scale = (budget / float(src_w * src_h)) ** 0.5
    width = max(16, int(round(src_w * scale / 16)) * 16)
    height = max(16, int(round(src_h * scale / 16)) * 16)
    return width, height


# ── media preparation ────────────────────────────────────────────────────────


def probe_video(path: str) -> tuple[int, float, int, int]:
    """(frames, fps, width, height) for a driving clip."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise BackendError(f"Could not open {os.path.basename(path)}.")
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if frames <= 0 or width <= 0:
        raise BackendError(
            f"{os.path.basename(path)} has no readable video track.")
    return frames, (fps if fps > 1e-3 else 30.0), width, height


def cover_resize(image: numpy.ndarray, width: int, height: int) -> numpy.ndarray:
    """Scale to fill and centre-crop. No bars: the model treats them as content."""
    src_h, src_w = image.shape[:2]
    scale = max(width / src_w, height / src_h)
    resized = cv2.resize(
        image, (max(width, int(round(src_w * scale))),
                max(height, int(round(src_h * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    top = (resized.shape[0] - height) // 2
    left = (resized.shape[1] - width) // 2
    return resized[top:top + height, left:left + width]


def resample_driving(path: str, dest: str, settings: TransferSettings,
                     should_cancel: Optional[Callable[[], bool]] = None) -> int:
    """Rewrite the driving clip at the target size and frame rate.

    Doing the geometry here rather than in the graph keeps the backend's decode
    deterministic and lets a long clip be cut into per-pass pieces without
    holding every frame of it in memory at once.
    """
    frames, fps, _w, _h = probe_video(path)
    step = fps / float(settings.fps)
    capture = cv2.VideoCapture(path)
    writer = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*"mp4v"),
                             settings.fps, (settings.width, settings.height))
    if not writer.isOpened():
        capture.release()
        raise BackendError("Could not open a writer for the prepared clip.")
    limit = int(settings.max_seconds * settings.fps) if settings.max_seconds else 0
    written = 0
    position = 0.0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= position:
                writer.write(cover_resize(frame, settings.width, settings.height))
                written += 1
                position += step
                if limit and written >= limit:
                    break
            index += 1
            if should_cancel is not None and should_cancel():
                raise InterruptedError
    finally:
        capture.release()
        writer.release()
    if written == 0:
        raise BackendError("The driving clip produced no frames.")
    return written


def cut_segment(src: str, dest: str, start: int, count: int) -> int:
    """Write ``count`` frames from ``start``, holding the last frame if short."""
    capture = cv2.VideoCapture(src)
    if not capture.isOpened():
        raise BackendError("Could not reopen the prepared clip.")
    fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    written = 0
    last = None
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        while written < count:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            last = frame
            written += 1
        # The model needs a full block; upstream pads by holding the final frame.
        while written < count and last is not None:
            writer.write(last)
            written += 1
    finally:
        capture.release()
        writer.release()
    if written == 0:
        raise BackendError(f"No driving frames available at frame {start}.")
    return written


# ── the graph ────────────────────────────────────────────────────────────────


def build_prompt(reference_name: str, pose_name: str, segment: Segment,
                 settings: TransferSettings, seed: int,
                 continue_name: Optional[str], prefix: str) -> dict:
    """Assemble the API-format graph for a single generation pass.

    Mirrors the reference workflow's wiring, minus its resize nodes: the media
    is already at the target size, so width and height are literals here.
    """
    graph: dict[str, dict] = {
        "unet": {"class_type": "UNETLoader", "inputs": {
            "unet_name": UNET_NAME, "weight_dtype": "default"}},
        "sampling": {"class_type": "ModelSamplingSD3", "inputs": {
            "model": ["unet", 0], "shift": SHIFT}},
        "clip": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": CLIP_NAME, "type": "wan", "device": "default"}},
        "positive": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["clip", 0], "text": settings.prompt}},
        "negative": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["clip", 0], "text": NEGATIVE_PROMPT}},
        "pose_text": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["clip", 0], "text": settings.pose_prompt}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "clip_vision": {"class_type": "CLIPVisionLoader", "inputs": {
            "clip_name": CLIP_VISION_NAME}},
        "reference": {"class_type": "LoadImage", "inputs": {
            "image": reference_name}},
        "reference_cv": {"class_type": "CLIPVisionEncode", "inputs": {
            "clip_vision": ["clip_vision", 0], "image": ["reference", 0],
            "crop": "none"}},
        "pose_video": {"class_type": "LoadVideo", "inputs": {"file": pose_name}},
        "pose_frames": {"class_type": "GetVideoComponents", "inputs": {
            "video": ["pose_video", 0]}},
        "pose_first": {"class_type": "ImageFromBatch", "inputs": {
            "image": ["pose_frames", 0], "batch_index": 0, "length": 1}},
        "pose_cv": {"class_type": "CLIPVisionEncode", "inputs": {
            "clip_vision": ["clip_vision", 0], "image": ["pose_first", 0],
            "crop": "none"}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {
            "sampler_name": SAMPLER}},
        "sigmas": {"class_type": "BasicScheduler", "inputs": {
            "model": ["sampling", 0], "scheduler": SCHEDULER,
            "steps": settings.steps, "denoise": 1.0}},
    }

    # The cache runs the pose branch once per pass instead of once per step --
    # roughly a 2x speedup bought with memory. Turning it *off* means leaving
    # the node out: its "default" dtype is the largest setting, not the absent
    # one, so routing "no cache" through it would ask for the most memory of
    # all rather than the least.
    if settings.cache_dtype != CACHE_OFF:
        graph["cache"] = {"class_type": "WanAnimate2Cache", "inputs": {
            "model": ["unet", 0], "device": settings.cache_device,
            "dtype": settings.cache_dtype}}
        graph["sampling"]["inputs"]["model"] = ["cache", 0]

    animate_inputs = {
        "positive": ["positive", 0],
        "negative": ["negative", 0],
        "vae": ["vae", 0],
        "width": settings.width,
        "height": settings.height,
        "length": segment.length,
        "batch_size": 1,
        "reference_image": ["reference", 0],
        "pose_video": ["pose_frames", 0],
        "clip_vision_output": ["reference_cv", 0],
        "positive_pose": ["pose_text", 0],
        "clip_vision_output_pose": ["pose_cv", 0],
        # The clip handed to this pass already starts at the right place, so the
        # node only needs to know whether an anchor frame precedes the new work.
        "video_frame_offset": 1 if segment.trim_first else 0,
        "pose_strength": settings.pose_strength,
        "pose_start_percent": 0.0,
        "pose_end_percent": 1.0,
        "reference_image_strength": settings.reference_strength,
    }
    if continue_name:
        graph["continue"] = {"class_type": "LoadImage",
                             "inputs": {"image": continue_name}}
        animate_inputs["continue_motion"] = ["continue", 0]

    graph["animate"] = {"class_type": "WanAnimate2ToVideo",
                        "inputs": animate_inputs}
    graph["sample"] = {"class_type": "SamplerCustom", "inputs": {
        "model": ["sampling", 0], "add_noise": True, "noise_seed": seed,
        "cfg": CFG, "positive": ["animate", 0], "negative": ["animate", 1],
        "sampler": ["sampler", 0], "sigmas": ["sigmas", 0],
        "latent_image": ["animate", 2]}}
    graph["trim"] = {"class_type": "TrimVideoLatent", "inputs": {
        "samples": ["sample", 0], "trim_amount": ["animate", 3]}}
    graph["decode"] = {"class_type": "VAEDecode", "inputs": {
        "samples": ["trim", 0], "vae": ["vae", 0]}}

    source = ["decode", 0]
    if segment.trim_first:
        graph["drop_anchor"] = {"class_type": "ImageFromBatch", "inputs": {
            "image": ["decode", 0], "batch_index": ["animate", 4],
            "length": 4096}}
        source = ["drop_anchor", 0]
    graph["save"] = {"class_type": "SaveImage", "inputs": {
        "images": source, "filename_prefix": prefix}}
    return graph


# ── the backend process ──────────────────────────────────────────────────────


def backend_paths() -> dict:
    root = BACKEND_ROOT
    return {
        "root": root,
        "comfy": os.path.join(root, "ComfyUI"),
        "python": os.path.join(root, "venv", "Scripts", "python.exe"),
        "input": os.path.join(root, "input"),
        "output": os.path.join(root, "output"),
        "temp": os.path.join(root, "temp"),
        "user": os.path.join(root, "user"),
    }


def backend_installed() -> bool:
    paths = backend_paths()
    return (os.path.isfile(os.path.join(paths["comfy"], "main.py"))
            and os.path.isfile(paths["python"]))


def models_dir() -> str:
    """The shared models folder the backend reads."""
    config = os.path.join(backend_paths()["comfy"], "extra_model_paths.yaml")
    try:
        with open(config, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip().startswith("base_path:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return os.path.join(backend_paths()["comfy"], "models")


def safetensors_complete(path: str) -> bool:
    """True if the file is a whole safetensors archive, not a partial download.

    ``os.path.isfile`` is happy with a file that is still being written, which
    would let a render start and then fail deep into the expensive part. The
    format declares its own size -- an 8-byte header length, that many bytes of
    JSON, then tensor data at offsets the JSON lists -- so the complete size can
    be derived and compared instead of hardcoding numbers that upstream may
    change.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            raw = handle.read(8)
            if len(raw) < 8:
                return False
            header_length = int.from_bytes(raw, "little")
            if header_length <= 0 or header_length > size:
                return False
            header = json.loads(handle.read(header_length).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    end = 0
    for key, entry in header.items():
        if key == "__metadata__" or not isinstance(entry, dict):
            continue
        offsets = entry.get("data_offsets")
        if isinstance(offsets, (list, tuple)) and len(offsets) == 2:
            end = max(end, int(offsets[1]))
    return size >= 8 + header_length + end


def missing_models() -> list[str]:
    """Model files that are absent or still downloading."""
    base = models_dir()
    missing = []
    for folder, name in MODEL_FILES.items():
        path = os.path.join(base, folder, name)
        if not os.path.isfile(path) or not safetensors_complete(path):
            missing.append(name)
    return missing


def model_available() -> bool:
    return backend_installed() and not missing_models()


class Backend:
    """Starts the private ComfyUI on demand and keeps it warm between renders.

    Loading a 15 GB checkpoint takes long enough that tearing the process down
    after every render would dominate the cost of a short clip, so the server is
    left running once started and reused.
    """

    def __init__(self, host: str = BACKEND_HOST, port: int = BACKEND_PORT):
        self.host = host
        self.port = port
        self.client_id = str(uuid.uuid4())
        self._process: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def alive(self, timeout: float = 2.0) -> bool:
        try:
            with urllib.request.urlopen(
                    f"{self.base_url}/system_stats", timeout=timeout):
                return True
        except Exception:
            return False

    def ensure(self, timeout: float = 240.0,
               progress: Optional[Callable[[str, float], None]] = None) -> None:
        if self.alive():
            return
        if not backend_installed():
            raise BackendError(
                "The Wan-Animate backend is not installed. Expected it at "
                f"{BACKEND_ROOT}.")
        paths = backend_paths()
        for key in ("input", "output", "temp", "user"):
            os.makedirs(paths[key], exist_ok=True)
        command = [
            paths["python"], "main.py",
            "--port", str(self.port),
            "--disable-auto-launch",
            "--input-directory", paths["input"],
            "--output-directory", paths["output"],
            "--temp-directory", paths["temp"],
            "--user-directory", paths["user"],
        ] + BACKEND_MEMORY_FLAGS
        creation = 0
        if sys.platform == "win32":
            creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        log = open(os.path.join(paths["root"], "server.log"), "ab", buffering=0)
        self._process = subprocess.Popen(
            command, cwd=paths["comfy"], stdout=log, stderr=log,
            creationflags=creation)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.alive(timeout=1.5):
                return
            if self._process.poll() is not None:
                raise BackendError(
                    "The render backend exited on startup. See "
                    f"{os.path.join(paths['root'], 'server.log')}.")
            if progress is not None:
                progress("Starting the render backend...", 0.0)
            time.sleep(2.0)
        raise BackendError("The render backend did not come up in time.")

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=15)
            except Exception:
                self._process.kill()
        self._process = None

    def free_memory(self) -> None:
        """Ask the backend to drop models -- used before giving the GPU back."""
        try:
            request = urllib.request.Request(
                f"{self.base_url}/free", method="POST",
                data=json.dumps({"unload_models": True,
                                 "free_memory": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(request, timeout=30).read()
        except Exception:
            pass

    # -- job submission --

    def submit(self, graph: dict) -> str:
        payload = json.dumps({"prompt": graph,
                              "client_id": self.client_id}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/prompt", data=payload, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())["prompt_id"]
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:2000]
            raise BackendError(f"The backend rejected the job: {detail}") from error

    def history(self, prompt_id: str) -> dict:
        with urllib.request.urlopen(
                f"{self.base_url}/history/{prompt_id}", timeout=30) as response:
            return json.loads(response.read()).get(prompt_id, {})

    def interrupt(self) -> None:
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"{self.base_url}/interrupt", method="POST", data=b""),
                timeout=15).read()
        except Exception:
            pass

    def wait(self, prompt_id: str,
             on_step: Optional[Callable[[int, int], None]] = None,
             should_cancel: Optional[Callable[[], bool]] = None) -> dict:
        """Block until the job finishes, reporting sampler steps as they land."""
        import websocket  # deferred: only motion transfer needs it

        query = urllib.parse.urlencode({"clientId": self.client_id})
        socket = websocket.WebSocket()
        socket.connect(f"ws://{self.host}:{self.port}/ws?{query}", timeout=30)
        socket.settimeout(2.0)
        try:
            while True:
                if should_cancel is not None and should_cancel():
                    self.interrupt()
                    raise InterruptedError
                try:
                    message = socket.recv()
                except Exception:
                    # A quiet socket is normal: a single step can take minutes.
                    if self._finished(prompt_id):
                        break
                    continue
                if not isinstance(message, str):
                    continue
                event = json.loads(message)
                kind, data = event.get("type"), event.get("data", {})
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                if kind == "progress" and on_step is not None:
                    on_step(int(data.get("value", 0)), int(data.get("max", 1)))
                elif kind == "execution_error":
                    raise BackendError(
                        f"Render failed: {data.get('exception_message', '')}")
                elif kind == "executing" and data.get("node") is None:
                    break
                elif kind == "execution_success":
                    break
        finally:
            try:
                socket.close()
            except Exception:
                pass
        return self._settled(prompt_id)

    def _settled(self, prompt_id: str, timeout: float = 60.0) -> dict:
        """Read the history once the outputs are actually in it.

        The socket announces the end of execution slightly before the history
        entry is written, so reading it immediately can come back empty -- which
        looks exactly like a render that produced nothing, after quarter of an
        hour of work that in fact succeeded.
        """
        deadline = time.time() + timeout
        record: dict = {}
        while time.time() < deadline:
            try:
                record = self.history(prompt_id)
            except Exception:
                record = {}
            if _saved_images(record):
                return record
            if record.get("status", {}).get("status_str") == "error":
                return record
            time.sleep(0.5)
        return record

    def _finished(self, prompt_id: str) -> bool:
        try:
            record = self.history(prompt_id)
        except Exception:
            return False
        return bool(_saved_images(record)) or bool(
            record.get("status", {}).get("completed"))


# ── running a job ────────────────────────────────────────────────────────────


#: Key of the node whose output is the rendered frames. Named rather than
#: searched for: LoadVideo also reports "images" (a one-frame preview), so
#: taking the first node that has any would silently return a single frame.
SAVE_NODE = "save"


def _saved_images(record: dict, node: str = SAVE_NODE) -> list[dict]:
    output = record.get("outputs", {}).get(node) or {}
    return output.get("images", [])


def _image_path(entry: dict) -> str:
    root = backend_paths()["output"]
    if entry.get("type") == "temp":
        root = backend_paths()["temp"]
    return os.path.join(root, entry.get("subfolder", ""), entry["filename"])


@dataclass
class TransferResult:
    path: str
    frames: int
    seconds: float
    segments: int
    minutes_per_segment: float = 0.0
    #: passes that finished. Below ``segments`` when a render was stopped.
    completed_segments: int = 0
    #: frames the model actually generated, anchors included
    generated_frames: int = 0
    #: why it stopped early, empty if it did not
    error: str = ""

    @property
    def complete(self) -> bool:
        return not self.error


def transfer(reference: str, driving: str, output: str,
             settings: Optional[TransferSettings] = None,
             progress: Optional[Callable[[str, float], None]] = None,
             should_cancel: Optional[Callable[[], bool]] = None,
             backend: Optional[Backend] = None) -> TransferResult:
    """Render ``reference``'s character performing ``driving``'s motion.

    Covers the whole driving clip regardless of length by chaining as many
    generation passes as it takes.
    """
    settings = (settings or TransferSettings()).snapped()
    report = progress or (lambda _m, _f: None)
    cancelled = should_cancel or (lambda: False)

    if not model_available():
        missing = ", ".join(missing_models()) or "the backend"
        raise BackendError(f"Not ready to render: missing {missing}.")

    backend = backend or Backend()
    backend.ensure(progress=report)

    paths = backend_paths()
    job = f"morphify_{int(time.time())}_{os.getpid()}"
    work = os.path.join(paths["input"], job)
    os.makedirs(work, exist_ok=True)
    seed = settings.seed
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")

    try:
        report("Preparing the driving clip...", 0.01)
        prepared = os.path.join(work, "drive.mp4")
        total = resample_driving(driving, prepared, settings, cancelled)

        reference_image = cv2.imread(reference, cv2.IMREAD_COLOR)
        if reference_image is None:
            raise BackendError(
                f"Could not read {os.path.basename(reference)}.")
        reference_name = f"{job}/reference.png"
        cv2.imwrite(os.path.join(work, "reference.png"),
                    cover_resize(reference_image, settings.width,
                                 settings.height))

        segments = plan_segments(total)
        report(f"{total} frames -> {len(segments)} pass"
               f"{'es' if len(segments) != 1 else ''}.", 0.02)

        frames: list[str] = []
        continue_name: Optional[str] = None
        started = time.time()
        total_steps = max(1, len(segments) * settings.steps)
        done_steps = 0
        completed = 0
        generated = 0
        failure = ""

        for segment in segments:
            # A pass costs minutes, so a failure in a later one must not throw
            # away the earlier ones. Whatever finished gets written out below.
            try:
                if cancelled():
                    raise InterruptedError
                frames.extend(_run_pass(
                    backend, segment, segments, settings, seed, job, work,
                    prepared, reference_name, continue_name, done_steps,
                    total_steps, report, cancelled))
            except Exception as error:
                failure = ("Cancelled." if isinstance(error, InterruptedError)
                           else str(error))
                break
            done_steps += settings.steps
            completed += 1
            generated += segment.length
            # The next pass anchors on this pass's final frame.
            anchor_name = f"anchor_{segment.index:03d}.png"
            shutil.copyfile(frames[-1], os.path.join(work, anchor_name))
            continue_name = f"{job}/{anchor_name}"

        if not frames:
            raise BackendError(failure or "No frames were generated.")

        report("Writing the video...", 0.96)
        written = write_video(frames, output, settings.fps)
        if settings.output_fps > settings.fps:
            report(f"Smoothing to {settings.output_fps:.0f} fps...", 0.98)
            interpolate_video(output, settings.output_fps)
        _mux_audio(driving, output)

        elapsed = time.time() - started
        report("Done." if not failure else "Stopped early; saved what rendered.",
               1.0)
        return TransferResult(
            path=output, frames=written, seconds=elapsed,
            segments=len(segments), completed_segments=completed,
            generated_frames=generated,
            minutes_per_segment=(elapsed / 60.0) / max(1, completed),
            error=failure)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run_pass(backend: "Backend", segment: Segment, segments: list,
              settings: TransferSettings, seed: int, job: str, work: str,
              prepared: str, reference_name: str,
              continue_name: Optional[str], done_steps: int, total_steps: int,
              report: Callable[[str, float], None],
              cancelled: Callable[[], bool]) -> list[str]:
    """Render one pass and return the paths of the frames it produced."""
    clip = os.path.join(work, f"pose_{segment.index:03d}.mp4")
    cut_segment(prepared, clip, segment.read_start, segment.length)
    graph = build_prompt(
        reference_name, f"{job}/pose_{segment.index:03d}.mp4", segment,
        settings, seed, continue_name,
        prefix=f"{job}/seg{segment.index:03d}")

    def on_step(value: int, maximum: int) -> None:
        report(f"Pass {segment.index + 1} of {len(segments)}  -  "
               f"step {value}/{maximum}",
               0.02 + 0.93 * ((done_steps + value) / total_steps))

    prompt_id = backend.submit(graph)
    record = backend.wait(prompt_id, on_step=on_step, should_cancel=cancelled)
    images = _saved_images(record)
    if not images:
        raise BackendError(f"Pass {segment.index + 1} produced no frames.")
    return [_image_path(entry) for entry in images]


def write_video(frames: Iterable[str], output: str, fps: float) -> int:
    """Encode the generated frames to an H.264 mp4.

    OpenCV's ``mp4v`` writes MPEG-4 Part 2, which phones and messaging apps
    refuse outright -- WhatsApp rejected the first clip made here. Frames are
    piped to ffmpeg instead and encoded as H.264 / yuv420p with faststart,
    which is the combination that plays everywhere. OpenCV stays as a fallback
    for the case where ffmpeg is not installed.
    """
    frames = list(frames)
    if not frames:
        raise BackendError("No frames were generated.")
    first = cv2.imread(frames[0], cv2.IMREAD_COLOR)
    if first is None:
        raise BackendError("The generated frames could not be read back.")
    height, width = first.shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        written = _encode_h264(ffmpeg, frames, output, fps, width, height)
        if written:
            return written
    return _encode_opencv(frames, output, fps, width, height)


def _read_sized(path: str, width: int, height: int):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height))
    return image


def _encode_h264(ffmpeg: str, frames: list[str], output: str, fps: float,
                 width: int, height: int) -> int:
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", output,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    written = 0
    try:
        for path in frames:
            image = _read_sized(path, width, height)
            if image is None:
                continue
            process.stdin.write(image.tobytes())
            written += 1
    except (BrokenPipeError, OSError):
        written = 0
    finally:
        try:
            process.stdin.close()
        except Exception:
            pass
        process.wait()
    if process.returncode != 0 or not os.path.isfile(output):
        return 0
    return written


def _encode_opencv(frames: list[str], output: str, fps: float,
                   width: int, height: int) -> int:
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    if not writer.isOpened():
        raise BackendError(f"Could not write {output}.")
    written = 0
    try:
        for path in frames:
            image = _read_sized(path, width, height)
            if image is None:
                continue
            writer.write(image)
            written += 1
    finally:
        writer.release()
    if not os.path.isfile(output):
        raise BackendError(f"{output} was not created.")
    return written


def interpolate_video(video: str, target_fps: float) -> bool:
    """Raise a clip's frame rate by synthesising the in-between frames.

    The expensive part of this pipeline is generating frames, and that cost is
    per frame regardless of what rate they are played at. So generating at 8 fps
    and interpolating up to 24 covers three times as much footage for the same
    work. Measured at roughly 26 seconds for a 7-second clip, against renders
    measured in tens of minutes -- about 1% overhead to halve the render.

    Motion-compensated interpolation invents frames, so it is a real trade: fast
    motion can smear. It is offered as a choice rather than applied by default.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or target_fps <= 0:
        return False
    smoothed = video + ".smooth.mp4"
    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", video,
         "-vf", (f"minterpolate=fps={target_fps:g}:mi_mode=mci:"
                 "mc_mode=aobmc:me_mode=bidir:vsbmc=1"),
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", smoothed],
        capture_output=True)
    if result.returncode == 0 and os.path.isfile(smoothed):
        os.replace(smoothed, video)
        return True
    try:
        os.remove(smoothed)
    except OSError:
        pass
    return False


def _mux_audio(source: str, video: str) -> None:
    """Carry the driving clip's audio over, if ffmpeg and audio are present."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return
    merged = video + ".audio.mp4"
    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", video, "-i", source,
         "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
         "-shortest", merged],
        capture_output=True)
    if result.returncode == 0 and os.path.isfile(merged):
        os.replace(merged, video)
    else:
        try:
            os.remove(merged)
        except OSError:
            pass


# ── estimation ───────────────────────────────────────────────────────────────

#: Measured on the target machine and written back by ``record_measurement``.
#: Seeded from the first real run rather than guessed; see MORPHIFY.md.
_CALIBRATION_FILE = os.path.join(
    os.path.expanduser("~"), ".morphify_wan_calibration.json")
_DEFAULT_SECONDS_PER_STEP = 0.0  # unknown until measured


def _calibration() -> dict:
    try:
        with open(_CALIBRATION_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def work_units(settings: TransferSettings, generated_frames: int) -> float:
    """A size-independent measure of how much there is to do.

    The transformer runs over every frame of a pass at once, so cost scales with
    frames as well as with pixels and steps. Leaving frames out makes a short
    final pass look like a full one: a 33-frame render was recorded as if it
    predicted an 81-frame one and under-estimated it by roughly three times.
    """
    megapixels = (settings.width * settings.height) / 1_000_000.0
    return max(1e-6, settings.steps * megapixels * max(1, generated_frames))


def record_measurement(settings: TransferSettings, generated_frames: int,
                       seconds: float) -> None:
    """Store what a finished render cost, per step per megapixel per frame."""
    data = _calibration()
    data["seconds_per_unit"] = seconds / work_units(settings, generated_frames)
    data["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data["sample"] = {"width": settings.width, "height": settings.height,
                      "steps": settings.steps, "frames": generated_frames,
                      "seconds": round(seconds, 1)}
    try:
        with open(_CALIBRATION_FILE, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
    except OSError:
        pass


def planned_frames(settings: TransferSettings, driving_frames: int) -> int:
    """Frames the model will generate, including anchors it later discards."""
    frames = driving_frames
    if settings.max_seconds:
        frames = min(frames, int(settings.max_seconds * settings.fps))
    return sum(s.length for s in plan_segments(max(1, frames)))


def estimate_minutes(settings: TransferSettings,
                     driving_frames: int) -> Optional[tuple[float, float]]:
    """Predicted range, or None until a real render has been measured."""
    rate = _calibration().get("seconds_per_unit", _DEFAULT_SECONDS_PER_STEP)
    if not rate:
        return None
    seconds = rate * work_units(
        settings, planned_frames(settings, driving_frames))
    return (seconds / 60.0 * 0.8, seconds / 60.0 * 1.35)
