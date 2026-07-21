"""JSONL persistence, aggregation, and batching-knee detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .timing import summarize


IDENTITY_FIELDS = (
    "experiment",
    "hardware_id",
    "device",
    "dtype",
    "attention",
    "use_cache",
    "quantization",
    "context_length",
    "batch_size",
    "output_tokens",
    "repetition",
)


def result_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in IDENTITY_FIELDS)


class JsonlStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSON on {self.path}:{line_number}") from error
        return rows

    def identities(self) -> set[tuple[Any, ...]]:
        return {result_identity(row) for row in self.read() if row.get("status") == "ok"}

    def append(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    group_fields = [field for field in IDENTITY_FIELDS if field != "repetition"]
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row.get(field) for field in group_fields)
        groups.setdefault(key, []).append(row)

    output = []
    for key, group in groups.items():
        aggregate = dict(zip(group_fields, key))
        for metric in ("ttft_seconds", "tpot_seconds", "tps", "end_to_end_seconds"):
            stats = summarize([float(row[metric]) for row in group])
            aggregate.update({f"{metric}_{name}": value for name, value in stats.items()})
        for metric in (
            "peak_rss_bytes",
            "peak_cuda_allocated_bytes",
            "peak_cuda_reserved_bytes",
        ):
            values = [row[metric] for row in group if row.get(metric) is not None]
            aggregate[f"{metric}_max"] = max(values) if values else None
        output.append(aggregate)
    return sorted(output, key=lambda row: tuple(str(row.get(field)) for field in group_fields))


def batching_knee(aggregates: Iterable[dict[str, Any]], threshold: float = 0.10) -> dict[str, Any] | None:
    ordered = sorted(aggregates, key=lambda row: int(row["batch_size"]))
    for previous, current in zip(ordered, ordered[1:]):
        previous_tps = float(previous["tps_p50"])
        gain = float(current["tps_p50"]) / previous_tps - 1.0
        latency_grew = (
            float(current["ttft_seconds_p95"]) > float(previous["ttft_seconds_p95"])
            or float(current["tpot_seconds_p95"]) > float(previous["tpot_seconds_p95"])
        )
        if gain < threshold and latency_grew:
            return {"batch_size": current["batch_size"], "marginal_tps_gain": gain}
    return None

