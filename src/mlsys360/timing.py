"""Timing primitives and metric calculations."""

from __future__ import annotations

import math
import time
from typing import Iterable, Sequence


def synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def timestamp(device: str) -> float:
    synchronize(device)
    return time.perf_counter()


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "mean": sum(data) / len(data) if data else math.nan,
        "p50": percentile(data, 0.50),
        "p95": percentile(data, 0.95),
        "p99": percentile(data, 0.99),
        "min": min(data) if data else math.nan,
        "max": max(data) if data else math.nan,
    }

