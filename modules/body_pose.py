"""Whole-body pose from a single image, via YOLOv8-pose in onnxruntime.

Seventeen COCO keypoints per person. Used by the puppet mode to learn where
a photographed person's limbs are, and where yours are, so one can be posed
like the other.

Kept separate from face_analyser: that model finds faces at five points and
knows nothing below the chin.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

from modules.paths import MODELS_DIR

MODEL_SUBDIR = "puppet"
MODEL_NAME = "yolov8n-pose.onnx"

INPUT_SIZE = 640

# COCO ordering, which every downstream index in the puppet rig assumes.
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

SKELETON = (
    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST), (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
)

CONFIDENCE = 0.25
_SESSION = None
_LOCK = threading.Lock()


def model_path() -> str:
    return os.path.join(MODELS_DIR, MODEL_SUBDIR, MODEL_NAME)


def model_available() -> bool:
    return os.path.isfile(model_path())


def _session():
    global _SESSION
    with _LOCK:
        if _SESSION is not None:
            return _SESSION
        import onnxruntime as ort

        from modules.processors.frame._onnx_enhancer import build_provider_config

        options = ort.SessionOptions()
        options.log_severity_level = 3
        _SESSION = ort.InferenceSession(
            model_path(), sess_options=options,
            providers=build_provider_config())
        return _SESSION


def unload() -> None:
    global _SESSION
    with _LOCK:
        _SESSION = None


class Pose:
    """One detected body: 17 keypoints with confidences, in image pixels."""

    __slots__ = ("points", "scores", "box", "score")

    def __init__(self, points: np.ndarray, scores: np.ndarray,
                 box: np.ndarray, score: float):
        self.points = points          # 17x2
        self.scores = scores          # 17
        self.box = box                # x1, y1, x2, y2
        self.score = score

    def visible(self, index: int, threshold: float = 0.35) -> bool:
        return bool(self.scores[index] >= threshold)

    def midpoint(self, a: int, b: int) -> np.ndarray:
        return (self.points[a] + self.points[b]) / 2.0

    @property
    def shoulder_width(self) -> float:
        return float(np.linalg.norm(
            self.points[LEFT_SHOULDER] - self.points[RIGHT_SHOULDER]))

    @property
    def torso_height(self) -> float:
        return float(np.linalg.norm(
            self.midpoint(LEFT_SHOULDER, RIGHT_SHOULDER)
            - self.midpoint(LEFT_HIP, RIGHT_HIP)))


def _letterbox(image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect, pad to a square. Returns scale and offsets."""
    height, width = image.shape[:2]
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    left = (INPUT_SIZE - new_w) // 2
    top = (INPUT_SIZE - new_h) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, scale, left, top


def detect(image: np.ndarray, confidence: float = CONFIDENCE) -> List[Pose]:
    """Every person in ``image`` (BGR), best first."""
    canvas, scale, left, top = _letterbox(image)
    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None]

    raw = _session().run(None, {"images": blob})[0]
    # (1, 56, 8400) -> (8400, 56): box(4) + score(1) + 17 keypoints x (x,y,c)
    predictions = raw[0].T

    keep = predictions[:, 4] >= confidence
    predictions = predictions[keep]
    if predictions.shape[0] == 0:
        return []

    boxes = predictions[:, :4].copy()
    # cx, cy, w, h -> x1, y1, x2, y2
    corners = np.empty_like(boxes)
    corners[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    corners[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    corners[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    corners[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), predictions[:, 4].tolist(), confidence, 0.45)
    if len(indices) == 0:
        return []
    indices = np.asarray(indices).reshape(-1)

    poses = []
    for index in indices:
        keypoints = predictions[index, 5:].reshape(17, 3)
        points = (keypoints[:, :2] - np.array([left, top])) / scale
        box = (corners[index] - np.array([left, top, left, top])) / scale
        poses.append(Pose(points.astype(np.float32),
                          keypoints[:, 2].astype(np.float32),
                          box.astype(np.float32),
                          float(predictions[index, 4])))
    poses.sort(key=lambda p: p.score, reverse=True)
    return poses


def detect_one(image: np.ndarray) -> Optional[Pose]:
    poses = detect(image)
    return poses[0] if poses else None


def draw(image: np.ndarray, pose: Pose, colour=(0, 255, 255)) -> np.ndarray:
    """Overlay the skeleton — for diagnostics, not the live output."""
    out = image.copy()
    for a, b in SKELETON:
        if pose.visible(a) and pose.visible(b):
            cv2.line(out, tuple(pose.points[a].astype(int)),
                     tuple(pose.points[b].astype(int)), colour, 2)
    for index in range(17):
        if pose.visible(index):
            cv2.circle(out, tuple(pose.points[index].astype(int)), 3,
                       (0, 0, 255), -1)
    return out
