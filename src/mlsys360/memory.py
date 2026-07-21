"""Peak CPU RSS and CUDA allocator memory measurement."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass


class RSSMonitor:
    def __init__(self, interval_seconds: float = 0.005) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        import psutil

        process = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                rss = process.memory_info().rss
                for child in process.children(recursive=True):
                    rss += child.memory_info().rss
                self.peak_bytes = max(self.peak_bytes, rss)
            except (psutil.Error, ProcessLookupError):
                pass
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "RSSMonitor":
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


@dataclass
class MemoryStats:
    peak_rss_bytes: int | None = None
    peak_cuda_allocated_bytes: int | None = None
    peak_cuda_reserved_bytes: int | None = None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
        }


def reset_cuda_peaks(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats(device)


def collect_memory(device: str, rss_peak: int) -> MemoryStats:
    if str(device).startswith("cuda"):
        import torch

        return MemoryStats(
            peak_rss_bytes=rss_peak,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    return MemoryStats(peak_rss_bytes=rss_peak)
