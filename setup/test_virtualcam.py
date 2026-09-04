"""End-to-end check of the virtual camera.

Publishes a known test pattern through ``VirtualCamSink``, then opens the
virtual camera as an ordinary capture device — the same way Discord or Zoom
would — and verifies the frames coming back match what was sent.

    python setup/test_virtualcam.py
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from modules.ui_common import get_available_cameras  # noqa: E402
from modules.virtual_camera import VirtualCamSink  # noqa: E402

WIDTH, HEIGHT, FPS = 1280, 720, 30

# Three flat colour bands, in BGR. Distinctive enough that a round-tripped
# frame can be identified even after NV12 chroma subsampling.
BANDS = [(32, 32, 220), (32, 200, 32), (220, 120, 32)]


def make_pattern() -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    band_h = HEIGHT // len(BANDS)
    for index, colour in enumerate(BANDS):
        frame[index * band_h:(index + 1) * band_h] = colour
    cv2.putText(frame, "DEEP LIVE CAM", (140, HEIGHT // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 8)
    return frame


def find_virtual_camera_index() -> int:
    _indices, names = get_available_cameras()
    for index, name in enumerate(names):
        if "virtual" in name.lower():
            return index
    return -1


def main() -> int:
    available, reason = VirtualCamSink.available()
    print(f"backend available : {available} ({reason})")
    if not available:
        return 1

    sink = VirtualCamSink()
    if not sink.start(WIDTH, HEIGHT, FPS):
        print(f"FAILED to open device: {sink.error}")
        return 1
    print(f"opened            : {sink.device_name} at {sink.resolution} @ {FPS}fps")

    pattern = make_pattern()
    stop = threading.Event()

    def pump() -> None:
        while not stop.is_set():
            sink.send(pattern)
            time.sleep(1.0 / FPS)

    feeder = threading.Thread(target=pump, daemon=True)
    feeder.start()
    time.sleep(1.5)  # let consumers see a steady stream before we attach

    index = find_virtual_camera_index()
    if index < 0:
        print("FAILED: no virtual camera in the device list")
        stop.set()
        sink.stop()
        return 1
    print(f"reading back from : device index {index}")

    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    ok = False
    received = None
    try:
        if not capture.isOpened():
            print("FAILED: could not open the virtual camera for reading")
        else:
            # Discard the first frames: the filter emits its idle placeholder
            # until it has picked up our stream.
            deadline = time.time() + 8
            while time.time() < deadline:
                got, frame = capture.read()
                if not got or frame is None:
                    continue
                received = frame
                if matches(frame):
                    ok = True
                    break
                time.sleep(0.05)
    finally:
        capture.release()
        stop.set()
        feeder.join(timeout=2)
        sink.stop()

    if received is not None:
        print(f"frame received    : {received.shape[1]}x{received.shape[0]}")
    print(f"frames sent       : {sink.frames_sent}")

    if ok:
        print("\nPASS - the published feed was read back as a camera device.")
        return 0
    print("\nFAIL - frames came back but did not match the test pattern.")
    return 1


def matches(frame: np.ndarray) -> bool:
    """Whether ``frame`` carries our three colour bands.

    Compares mean colour per band with a wide tolerance: the round trip goes
    through NV12, so exact values will not survive.
    """
    height, width = frame.shape[:2]
    if height < 90 or width < 160:
        return False
    band_h = height // len(BANDS)
    for index, expected in enumerate(BANDS):
        # Sample the middle of each band, away from the overlaid text.
        strip = frame[index * band_h + band_h // 6:
                      index * band_h + band_h // 4, :width // 6]
        if strip.size == 0:
            return False
        mean = strip.reshape(-1, 3).mean(axis=0)
        if np.abs(mean - np.array(expected, dtype=float)).max() > 60:
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
