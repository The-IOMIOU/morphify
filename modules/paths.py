"""Shared path constants for the Deep-Live-Cam project.

Running from a checkout, everything lives in the project folder, which is
what developers expect. Running from an installed build, the program folder
is read-only (Program Files), so anything the app writes goes to the
per-user data directory instead.

The models are the exception: they are ~1.5 GB, and a system drive is often
the one without room for them. The location is therefore overridable through
``config.json`` in the user data directory, which the Setup page edits. The
override is read once at import because the model paths are captured as
module-level constants by the frame processors — changing it takes effect on
the next launch, and the UI says so.
"""

import json
import os
import shutil
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IS_FROZEN = getattr(sys, "frozen", False)

if IS_FROZEN:
    # The bundle root is the folder holding the executable.
    ROOT_DIR = os.path.dirname(sys.executable)


APP_DIR_NAME = "Morphify"
# The app shipped as "DeepLiveCam" before the rename; settings written under
# that name are migrated once so nobody loses their models location or face
# library.
LEGACY_DIR_NAME = "DeepLiveCam"


def _user_data_dir(app_name: str = APP_DIR_NAME) -> str:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, app_name)
    if sys.platform == "darwin":
        return os.path.join(
            os.path.expanduser("~"), "Library", "Application Support", app_name)
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, app_name)


# Where the app keeps everything it writes.
USER_DATA_DIR = _user_data_dir() if IS_FROZEN else ROOT_DIR
LEGACY_USER_DATA_DIR = _user_data_dir(LEGACY_DIR_NAME) if IS_FROZEN else None


def migrate_legacy_data() -> bool:
    """Move a pre-rename data directory across, once. True if anything moved.

    Only runs for installed builds, and only when the new directory does not
    exist yet — never overwrites current data.
    """
    if not IS_FROZEN or not LEGACY_USER_DATA_DIR:
        return False
    if os.path.exists(USER_DATA_DIR) or not os.path.isdir(LEGACY_USER_DATA_DIR):
        return False
    try:
        os.rename(LEGACY_USER_DATA_DIR, USER_DATA_DIR)
        return True
    except OSError:
        # A cross-volume or locked directory: fall back to copying the small
        # settings files, which are the ones that actually hurt to lose.
        try:
            os.makedirs(USER_DATA_DIR, exist_ok=True)
            for name in ("config.json", "switch_states.json"):
                source = os.path.join(LEGACY_USER_DATA_DIR, name)
                if os.path.isfile(source):
                    shutil.copy2(source, os.path.join(USER_DATA_DIR, name))
            return True
        except OSError:
            return False


migrate_legacy_data()

CONFIG_PATH = os.path.join(USER_DATA_DIR, "config.json")
FACES_DIR = os.path.join(USER_DATA_DIR, "faces")
CAPTURES_DIR = os.path.join(USER_DATA_DIR, "captures")
SETTINGS_PATH = os.path.join(USER_DATA_DIR, "switch_states.json")

DEFAULT_MODELS_DIR = os.path.join(USER_DATA_DIR, "models")

# The full model set is about 1.6 GB on disk. A download also holds a
# ".part" copy of the file in flight, and the largest single model is
# ~0.55 GB, so the peak requirement is meaningfully higher than the final
# footprint. Rounded up to leave the volume some breathing room.
MODELS_REQUIRED_BYTES = 2_400_000_000


def read_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_config(config: dict) -> bool:
    try:
        os.makedirs(USER_DATA_DIR, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        return True
    except OSError:
        return False


def _resolve_models_dir() -> str:
    override = read_config().get("models_dir")
    if isinstance(override, str) and override.strip():
        return os.path.abspath(override)
    return DEFAULT_MODELS_DIR


MODELS_DIR = _resolve_models_dir()


def set_models_dir(path: str) -> bool:
    """Persist a new models location. Takes effect on the next launch."""
    config = read_config()
    config["models_dir"] = os.path.abspath(path)
    return write_config(config)


def free_bytes(path: str) -> int:
    """Free space on the volume holding ``path``, or -1 if unknown.

    Walks up to the nearest existing ancestor so a directory that has not
    been created yet still reports its future volume's free space.
    """
    probe = os.path.abspath(path)
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return -1
        probe = parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return -1


def has_room_for_models(path: str = None) -> bool:
    available = free_bytes(path or MODELS_DIR)
    return available < 0 or available >= MODELS_REQUIRED_BYTES


def ensure_user_dirs() -> None:
    """Create the writable directories. Safe to call repeatedly."""
    for path in (USER_DATA_DIR, MODELS_DIR, FACES_DIR, CAPTURES_DIR):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            pass
