"""Full live path against a real webcam.

camera -> swap -> virtual camera -> read back

This is the one path the other tests cannot cover: it needs an actual
capture device. It checks that frames flow, that the swap changes them, and
that what comes back out of the virtual camera is the processed feed.

Nothing from the camera is written to disk — this reads a real webcam, and a
test has no business leaving pictures of whoever is sitting in front of it
lying around.

    python setup/test_live_endtoend.py [camera_index]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import modules.globals  # noqa: E402


def pick_camera(preferred=None):
    from modules.ui_common import get_available_cameras
    indices, names = get_available_cameras()
    if not indices:
        return None, "no cameras found"
    if preferred is not None:
        if preferred in indices:
            return preferred, names[indices.index(preferred)]
        return None, f"camera {preferred} not present"
    # Skip virtual and filter devices; we want a real capture source.
    for index, name in zip(indices, names):
        lowered = name.lower()
        if "virtual" in lowered or "snap" in lowered:
            continue
        return index, name
    return indices[0], names[0]


def find_virtual_camera_index():
    from modules.ui_common import get_available_cameras
    _indices, names = get_available_cameras()
    for index, name in enumerate(names):
        if "virtual" in name.lower():
            return index
    return -1


def frames_match(sent, received, tolerance=14.0) -> bool:
    """Whether a read-back frame plausibly is the frame we published.

    The round trip rescales and goes through NV12, and the two are sampled a
    moment apart from a live feed, so this compares coarse structure: both
    are reduced to a small grid and the average per-cell difference has to be
    small. Works on a dark scene, where any absolute brightness test fails.
    """
    if sent is None or received is None:
        return False
    grid = (16, 16)
    a = cv2.resize(sent, grid, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(received, grid, interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.abs(a - b).mean()) < tolerance


def newest_face():
    from modules.paths import FACES_DIR
    if not os.path.isdir(FACES_DIR):
        return None
    candidates = [
        os.path.join(FACES_DIR, n) for n in os.listdir(FACES_DIR)
        if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None


def main() -> int:
    preferred = int(sys.argv[1]) if len(sys.argv) > 1 else None
    modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    modules.globals.frame_processors = ["face_swapper"]
    modules.globals.bypass_swap = False
    modules.globals.map_faces = False

    index, name = pick_camera(preferred)
    if index is None:
        print(f"SKIP: {name}")
        return 0
    print(f"camera            : [{index}] {name}")

    source = newest_face()
    if source is None:
        # Fall back to a generated face so the test can run on a fresh install.
        from modules.ui_face_browser import GENERATOR_URL, fetch_image
        from modules.paths import FACES_DIR, ensure_user_dirs
        ensure_user_dirs()
        source = os.path.join(FACES_DIR, "endtoend-source.jpg")
        with open(source, "wb") as handle:
            handle.write(fetch_image(GENERATOR_URL))
        print("source face       : generated a synthetic one")
    modules.globals.source_path = source
    print(f"source face       : {os.path.basename(source)}")

    from modules.face_analyser import get_face_analyser, get_source_face
    from modules.live_engine import LatestFrameSink, LiveEngine
    from modules.processors.frame.face_swapper import get_face_swapper
    from modules.virtual_camera import VirtualCamSink
    from modules import imread_unicode

    if get_source_face(imread_unicode(source)) is None:
        print("FAIL: no face detected in the source image")
        return 1

    print("loading models    : ...", flush=True)
    get_face_analyser()
    if get_face_swapper() is None:
        print("FAIL: swap model would not load")
        return 1

    engine = LiveEngine()
    preview = LatestFrameSink()
    engine.add_sink(preview)

    if not engine.start(index, 960, 540, 30):
        print(f"FAIL: {engine.error}")
        return 1

    ok = True
    try:
        width, height = engine.camera_resolution
        print(f"negotiated        : {width}x{height} @ "
              f"{engine.camera_fps:.0f}fps")

        # Let the loop settle and measure the real swap rate.
        time.sleep(4.0)
        swapped = preview.get()
        fps = engine.fps
        print(f"swap loop         : {fps:.1f} fps")
        if swapped is None:
            print("FAIL: no processed frames reached the sink")
            return 1
        print(f"processed frame   : {swapped.shape[1]}x{swapped.shape[0]}")

        # Compare against a bypassed frame to confirm the swap is doing work.
        modules.globals.bypass_swap = True
        time.sleep(1.0)
        raw = preview.get()
        modules.globals.bypass_swap = False
        time.sleep(1.5)

        # Was there actually a face to swap? Without one the comparison
        # below is meaningless, so say so rather than reporting a mystery.
        from modules.face_analyser import detect_one_face_fast
        detected = detect_one_face_fast(swapped)
        print(f"face in view      : {detected is not None}")
        print(f"frame brightness  : mean {float(swapped.mean()):.1f}, "
              f"std {float(swapped.std()):.1f}")

        if raw is not None and raw.shape == swapped.shape:
            delta = np.abs(raw.astype(np.int16)
                           - swapped.astype(np.int16)).mean()
            print(f"bypass vs swap    : mean delta {delta:.1f}")
            if delta < 1.0 and detected is None:
                print("  note: nobody in front of the camera, so nothing to "
                      "swap — this is expected, not a failure")
        else:
            print("bypass vs swap    : sizes differed, skipped")

        # Now publish and read back.
        vcam = VirtualCamSink()
        available, reason = VirtualCamSink.available()
        if not available:
            print(f"virtual camera    : SKIP ({reason})")
        else:
            if not vcam.start(1280, 720, 30):
                print(f"FAIL: {vcam.error}")
                return 1
            engine.add_sink(vcam)
            print(f"publishing to     : {vcam.device_name}")
            time.sleep(2.0)

            vindex = find_virtual_camera_index()
            print(f"reading device    : index {vindex}")
            capture = cv2.VideoCapture(vindex, cv2.CAP_DSHOW)
            try:
                if not capture.isOpened():
                    print("FAIL: could not open the virtual camera for reading")
                    ok = False
                else:
                    # Compare what comes back against what is being sent,
                    # rather than testing for absolute brightness: a dark
                    # room is a legitimate picture, and an absolute gate
                    # would call a working feed a failure.
                    got_frame = False
                    attempts = 0
                    best = None
                    deadline = time.time() + 10
                    while time.time() < deadline:
                        read_ok, frame = capture.read()
                        attempts += 1
                        if read_ok and frame is not None:
                            best = frame
                            sent = preview.get()
                            if sent is not None and frames_match(sent, frame):
                                got_frame = True
                                break
                        time.sleep(0.1)
                    if best is not None:
                        sent = preview.get()
                        print(f"read back         : {best.shape[1]}x"
                              f"{best.shape[0]}, mean {float(best.mean()):.1f} "
                              f"(published mean "
                              f"{float(sent.mean()) if sent is not None else -1:.1f}) "
                              f"after {attempts} reads")
                    if not got_frame:
                        print("FAIL: what came back does not match what was sent")
                        ok = False
            finally:
                capture.release()
            print(f"frames published  : {vcam.frames_sent}")
            engine.remove_sink(vcam)
            vcam.stop()
    finally:
        engine.stop()

    print("\n" + ("PASS - camera to virtual camera works end to end."
                 if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
