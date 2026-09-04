"""End-to-end check for offline motion transfer.

Run this with nothing else using the GPU. It renders a very short, low-step
clip — enough to prove the whole chain works without spending fifteen
minutes to find out it does not.

    venv\\Scripts\\python.exe setup\\test_motion_transfer.py [reference.jpg] [driving.mp4]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules.globals  # noqa: E402

modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test")


def check_resources() -> bool:
    """Refuse to start a long job that is going to run out of memory."""
    ok = True
    try:
        import psutil
        available = psutil.virtual_memory().available / 1024 ** 3
        print(f"  system RAM free : {available:5.1f} GB   (need ~12)")
        if available < 10:
            print("    -> too low. The text encoder alone is about 10 GB.")
            ok = False
    except ImportError:
        pass

    try:
        import torch
        if not torch.cuda.is_available():
            print("  CUDA            : unavailable -> would run on CPU, hours")
            return False
        free, total = torch.cuda.mem_get_info()
        print(f"  video RAM free  : {free / 1024 ** 3:5.1f} GB of "
              f"{total / 1024 ** 3:.1f} GB   (need ~8)")
        if free / 1024 ** 3 < 7:
            print("    -> too low. Close any game or other GPU app first.")
            ok = False
    except ImportError:
        print("  torch           : not installed")
        ok = False
    return ok


def main() -> int:
    from modules import motion_transfer as mt

    reference = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        TEST_DIR, "person_scene.jpg")
    driving = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        TEST_DIR, "driving.mp4")

    print("motion transfer self-check\n")
    print(f"  reference       : {reference}")
    print(f"  driving video   : {driving}")
    print(f"  model present   : {mt.model_available()}  ({mt.model_dir()})")
    if not mt.model_available():
        print("\nThe Wan model is not downloaded. Run:")
        print("  python -c \"import modules.motion_transfer as m;"
              " m.download_model(print)\"")
        return 1
    for path in (reference, driving):
        if not os.path.isfile(path):
            print(f"\nmissing input: {path}")
            return 1

    if not check_resources():
        print("\nSKIPPED: not enough free memory to run a fair test.")
        return 0

    # Deliberately small: proving the chain, not the quality.
    settings = mt.TransferSettings(
        width=384, height=672, frames=17, steps=6, seed=1234)
    low, high = mt.estimate_minutes(settings)
    print(f"\n  test render     : {settings.width}x{settings.height}, "
          f"{settings.frames} frames, {settings.steps} steps "
          f"(~{low:.0f}-{high:.0f} min)\n")

    output = os.path.join(TEST_DIR, "motion_out.mp4")
    started = time.time()

    def progress(message: str, fraction: float) -> None:
        sys.stdout.write(f"\r  [{fraction * 100:5.1f}%] {message[:66]:<66}")
        sys.stdout.flush()

    try:
        mt.transfer(reference, driving, output, settings, progress=progress)
    except Exception as exc:
        print(f"\n\nFAIL: {exc}")
        return 1

    print()
    if not os.path.isfile(output) or os.path.getsize(output) < 1000:
        print("\nFAIL: no video was written.")
        return 1

    import cv2
    capture = cv2.VideoCapture(output)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = capture.read()
    capture.release()

    print(f"\n  wrote           : {output}")
    print(f"  size            : {os.path.getsize(output) / 1048576:.1f} MB, "
          f"{frames} frames")
    print(f"  first frame     : "
          f"{'ok' if ok else 'UNREADABLE'} "
          f"{frame.shape if ok else ''}")
    print(f"  elapsed         : {(time.time() - started) / 60:.1f} min")

    if not ok:
        return 1
    print("\nPASS - motion transfer produced a playable video.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
