# --- START OF FILE globals.py ---

import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

# Face Mapping Data
source_target_map: List[Dict[str, Any]] = [] # Stores detailed map for image/video processing
simple_map: Dict[str, Any] = {}             # Stores simplified map (embeddings/faces) for live/simple mode

# Paths
source_path: str | None = None
target_path: str | None = None
output_path: str | None = None

# Processing Options
frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False         # Process all detected faces with default source
map_faces: bool = False          # Use source_target_map or simple_map for specific swaps
poisson_blend: bool = False      # Enable Poisson Blending for smoother face swaps
color_correction: bool = False   # Enable color correction (implementation specific)
nsfw_filter: bool = False

# Video Output Options
video_encoder: str | None = None
video_quality: int | None = None # Typically a CRF value or bitrate

# Live Mode Options
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None # Placeholder for UI element if needed
webcam_preview_running: bool = False
show_fps: bool = False
camera_width: int = 960          # Requested capture width (camera may override)
camera_height: int = 540         # Requested capture height
camera_fps: int = 60             # Requested capture rate

# Source identity
# Blend the ArcFace embedding across every photo sharing a name group
# (kai-cenat-01, kai-cenat-02, ...). Steadier identity than a single photo,
# and costs nothing per frame — the blend is computed once.
blend_identity: bool = True

# Full Takeover
# Beyond the face: wear the source's hair, skin tone and background too.
# Each stage is separate because they fail differently — hair struggles with
# big head turns, tone transfer with mismatched lighting, background with a
# cluttered room.
live_mode: str = "swap"        # which entry in ui.MODES is active
# Portrait animation drives a still with your webcam instead of swapping a
# face onto you. It replaces the swap rather than stacking with it.
# Withdrawn from the UI; see the note above ui.MODES.
portrait_enabled: bool = False
# Withdrawn: the hair transplant looked like a pasted cutout. The module
# remains for reference but the live loop never runs it.
takeover_enabled: bool = False
takeover_hair: bool = True
takeover_skin: bool = True
takeover_background: bool = True
takeover_skin_strength: float = 0.8   # 0 = leave your tone, 1 = fully theirs
takeover_hair_volume: float = 1.0     # yaw-driven shear that fakes hair depth

# Live pipeline tuning
# How often to re-run face detection, as a fraction of the camera's frame
# rate. Between detections the last result is reused, which is what keeps
# the swap real-time. Lower = smoother tracking, higher cost.
detect_interval_ratio: float = 0.08
# Passes the camera through untouched without stopping the stream — the
# "panic" control. Viewers keep seeing a feed; it is just your real face.
bypass_swap: bool = False
# Draws the original alongside the swapped output in the preview only.
split_view: bool = False
performance_preset: str = "Balanced"

# Virtual Camera Output
# Publishes the processed feed as a system camera device so other apps
# (Discord, Zoom, Teams, OBS, browsers) can consume the swap as a webcam.
virtual_cam_enabled: bool = False
virtual_cam_width: int = 1280
virtual_cam_height: int = 720
virtual_cam_fps: int = 30
# Mirroring is kept separate from live_mirror: the preview usually reads
# better mirrored (it matches a real mirror) while the outgoing feed should
# not be, or viewers see your text and gestures reversed.
virtual_cam_mirror: bool = False

# System Configuration
max_memory: int | None = None        # Memory limit in GB? (Needs clarification)
execution_providers: List[str] = []  # e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads: int | None = None # Number of threads for CPU execution
headless: bool | None = None         # Run without UI?
log_level: str = "error"             # Logging level (e.g., 'debug', 'info', 'warning', 'error')

# Face Processor UI Toggles (Example)
fp_ui: Dict[str, bool] = {"face_enhancer": False, "face_enhancer_gpen256": False, "face_enhancer_gpen512": False}

# Face Swapper Specific Options
face_swapper_enabled: bool = True # General toggle for the swapper processor
opacity: float = 1.0              # Blend factor for the swapped face (0.0-1.0)
# How far to push the identity past the halfway blend the swapper settles on.
# 0.0 is the model's own behaviour; higher looks more like the source face and
# eventually distorts. See face_swapper.strengthened_source.
identity_strength: float = 0.0
sharpness: float = 0.0            # Sharpness enhancement for swapped face (0.0-1.0+)

# Mouth Mask Options
mouth_mask: bool = False           # Enable mouth area masking/pasting
show_mouth_mask_box: bool = False  # Visualize the mouth mask area (for debugging)
mask_feather_ratio: int = 12       # Denominator for feathering calculation (higher = smaller feather)
mask_down_size: float = 0.1        # Expansion factor for lower lip mask (relative)
mask_size: float = 1.0             # Expansion factor for upper lip mask (relative)
mouth_mask_size: float = 0.0       # Mouth mask size (0-100; 0=off, 100=mouth to chin)

# --- START: Added for Frame Interpolation ---
enable_interpolation: bool = True # Toggle temporal smoothing
interpolation_weight: float = 0  # Blend weight for current frame (0.0-1.0). Lower=smoother.
# --- END: Added for Frame Interpolation ---

# --- END OF FILE globals.py ---

import threading
dml_lock = threading.Lock()
