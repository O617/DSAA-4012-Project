"""Small, dependency-light helpers for reproducible result provenance."""

from __future__ import annotations

import importlib.metadata
import platform


def software_versions() -> dict[str, str | None]:
    """Return the runtime versions needed to interpret benchmark results."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "accelerate", "datasets", "lm_eval", "torchao"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions
