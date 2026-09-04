"""Smoke test for the swap pipeline and the live engine's threading.

Runs without a webcam:

1. Downloads two synthetic faces, swaps one onto the other, and checks the
   output actually changed in the face region.
2. Drives ``LiveEngine`` with a stub capture device to confirm frames flow
   through the capture and processing threads out to every registered sink.

    python setup/test_pipeline.py
"""

import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import modules.globals  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test")


def fetch_face(name: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    if os.path.isfile(path) and os.path.getsize(path) > 1000:
        return path
    request = urllib.request.Request(
        "https://thispersondoesnotexist.com/random-person.jpeg",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
    if not data.startswith(b"\xff\xd8\xff"):
        raise RuntimeError(f"{name}: expected a JPEG, got {data[:16]!r}")
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def test_swap() -> bool:
    print("── swap pipeline ────────────────────────────────────────────")
    from modules import imread_unicode
    from modules.face_analyser import get_one_face, get_source_face
    from modules.processors.frame import face_swapper
    from modules.processors.frame.face_swapper import _has_ort_cuda

    print(f"  CUDA available   : {_has_ort_cuda()}")

    source_path = fetch_face("source.jpg")
    target_path = fetch_face("target.jpg")
    time.sleep(1)  # the endpoint rate-limits identical rapid requests

    source_img = imread_unicode(source_path)
    target_img = imread_unicode(target_path)
    if source_img is None or target_img is None:
        print("  FAIL: could not read the downloaded faces")
        return False

    # These are tight 1024x1024 headshots — the face fills the frame, which
    # RetinaFace cannot see without margin. get_source_face() retries with
    # padding; get_one_face() alone returns None, which is exactly the trap a
    # user hits when they pick an avatar as their source.
    print(f"  raw get_one_face : {get_one_face(source_img) is not None}")
    source_face = get_source_face(source_img)

    # The target is different: its bbox indexes back into the pixels we paste
    # into, so it has to be detected in the very image being swapped. Pad the
    # target first and keep that padded image as the target throughout.
    target_img = cv2.copyMakeBorder(
        target_img, 256, 256, 256, 256, cv2.BORDER_REPLICATE)
    target_face = get_one_face(target_img)

    if source_face is None or target_face is None:
        print("  FAIL: no face detected in the test images")
        return False
    print("  faces detected   : source + target")

    if face_swapper.get_face_swapper() is None:
        print("  FAIL: the swap model would not load")
        return False

    started = time.perf_counter()
    result = face_swapper.swap_face(source_face, target_face, target_img.copy())
    elapsed = time.perf_counter() - started
    print(f"  first swap       : {elapsed * 1000:.0f} ms (includes warm-up)")

    # Time a warm run — this is what the live loop actually costs.
    started = time.perf_counter()
    runs = 10
    for _ in range(runs):
        face_swapper.swap_face(source_face, target_face, target_img.copy())
    per_swap = (time.perf_counter() - started) / runs
    print(f"  warm swap        : {per_swap * 1000:.1f} ms "
          f"({1 / per_swap:.0f} swaps/sec)")

    x1, y1, x2, y2 = [int(v) for v in target_face.bbox]
    before = target_img[y1:y2, x1:x2].astype(np.int16)
    after = result[y1:y2, x1:x2].astype(np.int16)
    if before.size == 0:
        print("  FAIL: empty face region")
        return False
    delta = np.abs(after - before).mean()
    print(f"  mean pixel delta : {delta:.1f} inside the face box")

    cv2.imwrite(os.path.join(OUT_DIR, "swapped.png"), result)
    print(f"  wrote            : {os.path.join(OUT_DIR, 'swapped.png')}")

    if delta < 5:
        print("  FAIL: output is essentially identical to the target")
        return False
    print("  PASS")
    return True


class _StubCapture:
    """Stands in for VideoCapturer: emits a moving synthetic frame."""

    actual_width = 640
    actual_height = 360
    actual_fps = 30.0

    def __init__(self):
        self._n = 0

    def start(self, width=0, height=0, fps=0) -> bool:
        return True

    def read(self):
        frame = np.zeros((self.actual_height, self.actual_width, 3), dtype=np.uint8)
        offset = (self._n * 7) % self.actual_width
        frame[:, offset:offset + 40] = (40, 180, 240)
        self._n += 1
        time.sleep(1.0 / self.actual_fps)
        return True, frame

    def release(self):
        pass


def test_engine() -> bool:
    print("\n── live engine ──────────────────────────────────────────────")
    from modules import live_engine
    from modules.live_engine import LatestFrameSink, LiveEngine

    class CountingSink:
        def __init__(self):
            self.count = 0

        def send(self, frame):
            self.count += 1

    # No source face and no map: the loop runs the processor chain over each
    # frame and passes it through, which is all we need to prove the threads
    # and the sink fan-out work.
    modules.globals.source_path = None
    modules.globals.map_faces = False

    engine = LiveEngine()
    preview = LatestFrameSink()
    counter = CountingSink()
    engine.add_sink(preview)
    engine.add_sink(counter)

    original = live_engine.VideoCapturer
    live_engine.VideoCapturer = lambda index: _StubCapture()
    try:
        if not engine.start(0):
            print(f"  FAIL: engine did not start ({engine.error})")
            return False
        time.sleep(2.0)
        running = engine.is_running
        fps = engine.fps
        frame = preview.get()
        engine.stop()
    finally:
        live_engine.VideoCapturer = original

    print(f"  engine ran       : {running}")
    print(f"  frames to sink   : {counter.count}")
    print(f"  loop fps         : {fps:.1f}")
    print(f"  preview frame    : "
          f"{None if frame is None else f'{frame.shape[1]}x{frame.shape[0]}'}")
    print(f"  stopped clean    : {not engine.is_running}")

    if counter.count < 10 or frame is None:
        print("  FAIL: frames did not reach the sinks")
        return False
    print("  PASS")
    return True


def main() -> int:
    modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    modules.globals.execution_threads = 2
    modules.globals.frame_processors = ["face_swapper"]

    results = [test_swap(), test_engine()]
    print("\n" + ("ALL PASS" if all(results) else "FAILURES ABOVE"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
