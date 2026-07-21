"""Import guard for the vendored OpenHarness runtime."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


HARNESS_SRC = Path(__file__).resolve().parents[3] / "harness" / "src"


def _same_path(left: str, right: str) -> bool:
    return os.path.realpath(left) == os.path.realpath(right)


def _under_path(child: str, parent: str) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(child), os.path.realpath(parent)]) == os.path.realpath(parent)
    except ValueError:
        return False


def _module_uses_vendored_harness(module: ModuleType) -> bool:
    locations = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        locations.append(str(module_file))
    module_path = getattr(module, "__path__", None)
    if module_path:
        locations.extend(str(path) for path in module_path)
    return any(_under_path(location, str(HARNESS_SRC)) for location in locations)


def ensure_vendored_openharness_path() -> Path:
    """Pin `openharness` imports to this repository's vendored source tree."""
    if not HARNESS_SRC.exists():
        raise RuntimeError(f"Vendored OpenHarness source tree is missing: {HARNESS_SRC}")

    harness_src = str(HARNESS_SRC)
    sys.path[:] = [
        path for path in sys.path if not _same_path(path or os.getcwd(), harness_src)
    ]
    sys.path.insert(0, harness_src)

    loaded = sys.modules.get("openharness")
    if loaded is not None and not _module_uses_vendored_harness(loaded):
        for name in list(sys.modules):
            if name == "openharness" or name.startswith("openharness."):
                del sys.modules[name]
        importlib.invalidate_caches()

    return HARNESS_SRC


__all__ = ["HARNESS_SRC", "ensure_vendored_openharness_path"]
