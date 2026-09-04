#!/usr/bin/env python3

import os
import sys


# Add the project root to PATH so bundled ffmpeg/ffprobe are found
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] = project_root + os.pathsep + os.environ.get("PATH", "")

# Importing modules registers the CUDA/cuDNN library directories as a side
# effect (modules/gpu_paths.py), which has to happen before onnxruntime
# creates its first session. It used to be done inline here, which meant any
# other entry point silently lost GPU acceleration.
from modules import platform_info

# Intercepted before core.parse_args(), whose argparse rejects unknown flags.
if "--self-test" in sys.argv:
    from modules import self_test

    raise SystemExit(self_test.run())

platform_info.print_banner()

from modules import core

if __name__ == '__main__':
    core.run()
