"""First-run check for Wan-Animate-2 motion transfer.

Renders one short clip end to end and reports what it actually cost, which is
also what seeds the time estimate shown in the UI. Everything cheap is checked
first: a render that fails after twenty minutes because a game is holding the
GPU teaches nothing.

    venv\\Scripts\\python.exe setup\\test_wan_animate.py [reference.png] [drive.mp4]

With no arguments it generates a synthetic driving clip, which exercises the
whole path but tells you nothing about quality -- pass real files for that.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy  # noqa: E402

from modules import wan_animate as wa  # noqa: E402

GAME_VRAM_THRESHOLD_GB = 8.0

#: Below this, weights cannot be held at all and the run is pointless to time.
#: Between this and the full requirement it pages from disk: slower, not broken.
MINIMUM_RAM_GB = 8.0


def check_environment(settings) -> tuple[list[str], list[str]]:
    """(blocking problems, warnings), most important first."""
    problems: list[str] = []
    warnings: list[str] = []
    if not wa.backend_installed():
        problems.append(f"Backend missing at {wa.BACKEND_ROOT}.")
    missing = wa.missing_models() if wa.backend_installed() else []
    for name in missing:
        problems.append(f"Model not ready (absent or still downloading): {name}")

    want = wa.CHECKPOINT_GB + wa.CACHE_RAM_GB.get(settings.cache_dtype, 0.0)
    try:
        import psutil
        free_ram = psutil.virtual_memory().available / 1024 ** 3
        print(f"  system RAM free : {free_ram:5.1f} GB (this setup wants "
              f"~{want:.0f} GB)")
        if free_ram < MINIMUM_RAM_GB:
            problems.append(
                f"Only {free_ram:.1f} GB RAM free. Close some programs.")
        elif free_ram < want:
            warnings.append(
                f"{free_ram:.1f} GB free against ~{want:.0f} GB wanted -- "
                "weights will page from disk, so expect this to be slower than "
                "the same run on an idle machine.")
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            free_gb, total_gb = free / 1024 ** 3, total / 1024 ** 3
            print(f"  VRAM free       : {free_gb:5.1f} GB of {total_gb:.1f} GB")
            if free_gb < GAME_VRAM_THRESHOLD_GB:
                problems.append(
                    f"Only {free_gb:.1f} GB VRAM free -- something else (a game?) "
                    "is holding it. The result would be meaningless.")
        else:
            problems.append("CUDA is not available to torch.")
    except Exception as error:
        problems.append(f"Could not query the GPU: {error}")
    return problems, warnings


def synthetic_inputs(folder: str, width: int, height: int,
                     frames: int) -> tuple[str, str]:
    """A moving figure and a still reference -- enough to exercise the path."""
    os.makedirs(folder, exist_ok=True)
    reference = os.path.join(folder, "reference.png")
    canvas = numpy.full((height, width, 3), 60, numpy.uint8)
    cv2.circle(canvas, (width // 2, height // 3), width // 8, (200, 180, 170), -1)
    cv2.rectangle(canvas, (width // 3, height // 3 + width // 8),
                  (2 * width // 3, height - 40), (70, 90, 160), -1)
    cv2.imwrite(reference, canvas)

    drive = os.path.join(folder, "drive.mp4")
    writer = cv2.VideoWriter(drive, cv2.VideoWriter_fourcc(*"mp4v"), 16.0,
                             (width, height))
    for index in range(frames):
        frame = numpy.full((height, width, 3), 60, numpy.uint8)
        sway = int(numpy.sin(index / 6.0) * width / 10)
        cv2.circle(frame, (width // 2 + sway, height // 3), width // 8,
                   (200, 180, 170), -1)
        cv2.rectangle(frame, (width // 3 + sway, height // 3 + width // 8),
                      (2 * width // 3 + sway, height - 40), (70, 90, 160), -1)
        writer.write(frame)
    writer.release()
    return reference, drive


def main() -> int:
    print("Wan-Animate-2 motion transfer -- first-run check\n")
    print(f"  backend         : {wa.BACKEND_ROOT}")
    print(f"  models          : {wa.models_dir()}")

    # Nothing extra in RAM for the first measurement: the checkpoint alone does
    # not fit alongside a desktop on 24 GB, so the pose cache -- which is a
    # speed-for-memory trade -- is the wrong trade to make until the baseline is
    # known. It can be turned back on once there is a number to compare against.
    settings = wa.TransferSettings(width=480, height=848, steps=6,
                                   max_seconds=5.0,
                                   cache_dtype=wa.CACHE_OFF)

    problems, warnings = check_environment(settings)
    if problems:
        print("\nNot ready:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("  models          : all four present and complete")
    for warning in warnings:
        print(f"\n  note: {warning}")
    work = os.path.join(wa.backend_paths()["root"], "selftest")
    if len(sys.argv) >= 3:
        reference, drive = sys.argv[1], sys.argv[2]
        print(f"\n  reference       : {reference}")
        print(f"  driving         : {drive}")
    else:
        os.makedirs(work, exist_ok=True)
        reference, drive = synthetic_inputs(
            work, settings.width, settings.height, 81)
        print("\n  using a synthetic driving clip (pass two paths for a real one)")

    frames, fps, width, height = wa.probe_video(drive)
    print(f"  driving clip    : {frames} frames at {fps:.1f} fps, {width}x{height}")
    at_model_rate = int(frames * wa.DEFAULT_FPS / max(fps, 1e-3))
    capped = min(at_model_rate, int(settings.max_seconds * wa.DEFAULT_FPS))
    passes = wa.plan_segments(capped)
    print(f"  plan            : {capped} frames -> {len(passes)} pass"
          f"{'' if len(passes) == 1 else 'es'} at {settings.steps} steps")

    output = os.path.join(work, "result.mp4")
    os.makedirs(work, exist_ok=True)
    started = time.time()
    last = [0.0]

    def progress(message: str, fraction: float) -> None:
        now = time.time()
        if fraction >= 1.0 or now - last[0] > 5.0:
            last[0] = now
            print(f"    [{fraction * 100:5.1f}%] {message}")

    print("\nRendering. The first pass includes loading 15.5 GB, so it is the "
          "slow one.\n")
    try:
        result = wa.transfer(reference, drive, output, settings,
                             progress=progress)
    except Exception as error:
        print(f"\nFAIL: {error}")
        return 1

    minutes = result.seconds / 60.0
    print(f"\nPASS  {result.frames} frames -> {result.path}")
    print(f"      {minutes:.1f} min total, {result.minutes_per_segment:.1f} min "
          f"per pass, {len(passes)} pass(es)")
    wa.record_measurement(settings, result.generated_frames, result.seconds)

    window = wa.estimate_minutes(settings, at_model_rate)
    if window:
        print(f"      calibration stored; a full {frames / max(fps, 1):.0f} s "
              f"clip at these settings would be ~{window[0]:.0f}-{window[1]:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
