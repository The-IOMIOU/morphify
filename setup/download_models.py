"""Download the ONNX models the app needs.

Thin command-line front end over ``modules.model_store``, which holds the
catalogue so the app itself can fetch models on first launch too.

    python setup/download_models.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.model_store import (  # noqa: E402
    MODELS,
    MODELS_DIR,
    download,
    ensure_user_dirs,
    is_present,
    missing_models,
)


def _cli_progress(name: str, done: int, total: int) -> None:
    mb = done / 1_048_576
    if total:
        pct = 100 * done / total
        bar_width = 28
        filled = int(bar_width * done / total)
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stdout.write(f"\r  {name:<28} [{bar}] {pct:5.1f}%  {mb:7.1f} MB")
    else:
        sys.stdout.write(f"\r  {name:<28} {mb:7.1f} MB")
    sys.stdout.flush()


def main() -> int:
    ensure_user_dirs()
    print(f"Models directory: {MODELS_DIR}\n")

    for model in MODELS:
        tag = "required" if model.required else "optional"
        state = "present" if is_present(model) else "MISSING"
        print(f"  [{state:>7}] {model.filename:<28} ({tag}) - {model.purpose}")

    pending = missing_models()
    if not pending:
        print("\nAll models present.")
        return 0

    total_mb = sum(m.approx_bytes for m in pending) / 1_048_576
    print(f"\nDownloading {len(pending)} model(s), roughly {total_mb:.0f} MB...\n")

    failures = []
    for model in pending:
        if not download(model, progress=_cli_progress):
            failures.append(model)
        else:
            print(f"\r  {model.filename:<28} done" + " " * 30)

    if failures:
        print("\nFailed to download:")
        for model in failures:
            print(f"  - {model.filename} "
                  f"({'required' if model.required else 'optional'})")
            print(f"    {model.url}")
        if any(m.required for m in failures):
            print("\nA required model is missing; the app cannot swap faces "
                  "until it is downloaded.")
            return 1
        print("\nOnly optional models failed; the app will run without them.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
