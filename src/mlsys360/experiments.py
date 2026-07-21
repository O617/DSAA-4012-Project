"""Shared experiment runner used by all benchmark entry points."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterator

from .config import validate_config
from .decode import fixed_work_decode
from .model import load_model
from .prompts import exact_length_prompt
from .results import JsonlStore, aggregate_rows, batching_knee, result_identity


def _software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "accelerate", "torchao"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_state() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}


def workload_points(workload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if workload.get("points"):
        for point in workload["points"]:
            yield dict(point)
        return
    for context_length, batch_size in product(
        workload["context_lengths"], workload["batch_sizes"]
    ):
        yield {"context_length": int(context_length), "batch_size": int(batch_size)}


def variants(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    matrix = config.get("matrix", {})
    if not matrix:
        yield {}
        return
    keys = list(matrix)
    for values in product(*(matrix[key] for key in keys)):
        yield dict(zip(keys, values))


def _hardware_id(device: str) -> str:
    if str(device).startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            return f"{platform.node()}::{torch.cuda.get_device_name(device)}"
        return f"{platform.node()}::requested-cuda-unavailable"
    return f"{platform.node()}::{platform.processor() or platform.machine()}"


def _base_row(
    config: dict[str, Any], variant: dict[str, Any], point: dict[str, Any], repetition: int
) -> dict[str, Any]:
    runtime = config["runtime"]
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": config.get("experiment", "benchmark"),
        "hardware_id": _hardware_id(str(runtime["device"])),
        "device": runtime["device"],
        "dtype": runtime.get("dtype"),
        "attention": variant.get("attention", runtime.get("attention", "sdpa")),
        "use_cache": bool(variant.get("use_cache", runtime.get("use_cache", True))),
        "quantization": variant.get("quantization", "none"),
        "context_length": int(point["context_length"]),
        "batch_size": int(point["batch_size"]),
        "workload_label": point.get("label"),
        "output_tokens": int(config["workload"]["output_tokens"]),
        "repetition": repetition,
        "seed": int(config["workload"].get("seed", 4012)),
        "software": _software_versions(),
    }


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    validate_config(config)
    seed = int(config["workload"].get("seed", 4012))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    store = JsonlStore(config["output"]["path"])
    manifest_path = store.path.with_suffix(store.path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "software": _software_versions(),
                "git": _git_state(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    completed = store.identities() if config["output"].get("resume", True) else set()
    counts = {"completed": 0, "skipped": 0, "failed": 0}

    for variant in variants(config):
        variant_config = json.loads(json.dumps(config))
        variant_config["runtime"].update(
            {key: value for key, value in variant.items() if key in ("attention", "use_cache")}
        )
        quantization = str(variant.get("quantization", "none"))
        try:
            bundle = load_model(variant_config, quantization=quantization)
        except Exception as error:
            for point in workload_points(config["workload"]):
                row = _base_row(config, variant, point, -1)
                row.update(
                    status="configuration_error",
                    error_type=type(error).__name__,
                    error=str(error),
                )
                store.append(row)
                counts["failed"] += 1
            continue

        blocked_contexts: set[int] = set()
        for point in workload_points(config["workload"]):
            context_length = int(point["context_length"])
            if context_length in blocked_contexts:
                counts["skipped"] += int(config["workload"].get("repetitions", 1))
                continue
            try:
                prompt = exact_length_prompt(
                    bundle.tokenizer,
                    context_length,
                    int(point["batch_size"]),
                    bundle.device,
                )
                for _ in range(int(config["workload"].get("warmups", 0))):
                    fixed_work_decode(
                        bundle.model,
                        prompt.input_ids,
                        prompt.attention_mask,
                        int(config["workload"]["output_tokens"]),
                        bundle.device,
                        bool(variant_config["runtime"].get("use_cache", True)),
                    )

                for repetition in range(int(config["workload"].get("repetitions", 1))):
                    row = _base_row(config, variant, point, repetition)
                    if result_identity(row) in completed:
                        counts["skipped"] += 1
                        continue
                    result = fixed_work_decode(
                        bundle.model,
                        prompt.input_ids,
                        prompt.attention_mask,
                        int(config["workload"]["output_tokens"]),
                        bundle.device,
                        bool(variant_config["runtime"].get("use_cache", True)),
                    )
                    row.update(result.as_dict())
                    row.update(
                        status="ok",
                        prompt_hash=prompt.prompt_hash,
                        requested_output_tokens=int(config["workload"]["output_tokens"]),
                        actual_output_tokens=len(result.generated_token_ids[0]),
                        model=bundle.metadata,
                        quantization_details=bundle.quantization,
                    )
                    store.append(row)
                    completed.add(result_identity(row))
                    counts["completed"] += 1
            except torch.cuda.OutOfMemoryError as error:
                row = _base_row(config, variant, point, -1)
                row.update(status="oom", error_type=type(error).__name__, error=str(error))
                store.append(row)
                counts["failed"] += 1
                blocked_contexts.add(context_length)
                del prompt
                torch.cuda.empty_cache()
            except Exception as error:
                row = _base_row(config, variant, point, -1)
                row.update(status="error", error_type=type(error).__name__, error=str(error))
                store.append(row)
                counts["failed"] += 1

        try:
            del prompt
        except UnboundLocalError:
            pass
        del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = store.read()
    aggregates = aggregate_rows(rows)
    knees: dict[str, Any] = {}
    knee_groups = {
        (
            row["attention"],
            row["use_cache"],
            row["quantization"],
            row["context_length"],
        )
        for row in aggregates
    }
    for attention, use_cache, quantization, context_length in sorted(
        knee_groups, key=lambda values: tuple(map(str, values))
    ):
        subset = [
            row
            for row in aggregates
            if row["attention"] == attention
            and row["use_cache"] == use_cache
            and row["quantization"] == quantization
            and row["context_length"] == context_length
        ]
        key = f"{attention}|cache={use_cache}|{quantization}|context={context_length}"
        knees[key] = batching_knee(subset)
    return {**counts, "output": str(store.path), "batching_knees": knees}


def save_aggregates(raw_path: str | Path, output_path: str | Path) -> int:
    rows = JsonlStore(raw_path).read()
    aggregates = aggregate_rows(rows)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".json":
        destination.write_text(json.dumps(aggregates, indent=2) + "\n", encoding="utf-8")
    else:
        import pandas as pd

        pd.DataFrame(aggregates).to_csv(destination, index=False)
    return len(aggregates)
