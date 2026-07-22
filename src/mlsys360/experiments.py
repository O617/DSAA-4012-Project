"""Shared experiment runner used by all benchmark entry points."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterator

from .config import validate_config
from .decode import fixed_work_decode
from .model import load_model, local_model_artifact_hashes
from .prompts import exact_length_prompt
from .results import IDENTITY_FIELDS, JsonlStore, aggregate_rows, batching_knee, result_identity


def _software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "accelerate", "datasets", "lm_eval", "torchao"):
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


def _write_manifest(path: Path, config: dict[str, Any]) -> None:
    entry = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "software": _software_versions(),
        "git": _git_state(),
        "model_artifacts": local_model_artifact_hashes(str(config["model"]["model_id"])),
    }
    invocations: list[dict[str, Any]] = []
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid experiment manifest: {path}") from error
        if not isinstance(previous, dict):
            raise ValueError(f"Experiment manifest must contain a JSON object: {path}")
        if isinstance(previous.get("invocations"), list):
            invocations.extend(previous["invocations"])
        elif "config" in previous:
            invocations.append(
                {key: previous.get(key) for key in ("created_at", "config", "software", "git")}
            )
    payload = {"schema_version": 1, **entry, "invocations": [*invocations, entry]}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def _cpu_affinity() -> str | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    cpus = sorted(os.sched_getaffinity(0))
    ranges: list[str] = []
    start = previous = cpus[0]
    for cpu in cpus[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _base_row(
    config: dict[str, Any], variant: dict[str, Any], point: dict[str, Any], repetition: int
) -> dict[str, Any]:
    runtime = config["runtime"]
    model = config["model"]
    return {
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": config.get("experiment", "benchmark"),
        "hardware_id": _hardware_id(str(runtime["device"])),
        "model_id": model["model_id"],
        "model_revision": model.get("revision", "main"),
        "device": runtime["device"],
        "dtype": runtime.get("dtype"),
        "num_threads": runtime.get("num_threads"),
        "num_interop_threads": runtime.get("num_interop_threads"),
        "cpu_affinity": _cpu_affinity(),
        "attention": variant.get("attention", runtime.get("attention", "sdpa")),
        "use_cache": bool(variant.get("use_cache", runtime.get("use_cache", True))),
        "compile": bool(runtime.get("compile", False)),
        "quantization": variant.get("quantization", "none"),
        "context_length": int(point["context_length"]),
        "batch_size": int(point["batch_size"]),
        "workload_label": point.get("label"),
        "output_tokens": int(config["workload"]["output_tokens"]),
        "repetition": repetition,
        "seed": int(config["workload"].get("seed", 4012)),
        "software": _software_versions(),
        "logits_to_keep": 1,
    }


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    validate_config(config)
    runtime = config["runtime"]
    requested_threads = runtime.get("num_threads")
    if requested_threads is not None and torch.get_num_threads() != int(requested_threads):
        torch.set_num_threads(int(requested_threads))
    requested_interop_threads = runtime.get("num_interop_threads")
    if (
        requested_interop_threads is not None
        and torch.get_num_interop_threads() != int(requested_interop_threads)
    ):
        torch.set_num_interop_threads(int(requested_interop_threads))
    runtime["num_threads"] = torch.get_num_threads()
    runtime["num_interop_threads"] = torch.get_num_interop_threads()

    seed = int(config["workload"].get("seed", 4012))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    store = JsonlStore(config["output"]["path"])
    manifest_path = store.path.with_suffix(store.path.suffix + ".manifest.json")
    _write_manifest(manifest_path, config)
    completed = store.identities() if config["output"].get("resume", True) else set()
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    points = list(workload_points(config["workload"]))
    repetition_count = int(config["workload"].get("repetitions", 1))

    for variant in variants(config):
        variant_has_pending_work = any(
            result_identity(_base_row(config, variant, point, repetition)) not in completed
            for point in points
            for repetition in range(repetition_count)
        )
        if not variant_has_pending_work:
            counts["skipped"] += len(points) * repetition_count
            continue

        variant_config = json.loads(json.dumps(config))
        variant_config["runtime"].update(
            {key: value for key, value in variant.items() if key in ("attention", "use_cache")}
        )
        quantization = str(variant.get("quantization", "none"))
        try:
            bundle = load_model(variant_config, quantization=quantization)
        except Exception as error:
            for point in points:
                row = _base_row(config, variant, point, -1)
                row.update(
                    status="configuration_error",
                    error_type=type(error).__name__,
                    error=str(error),
                )
                store.append(row)
                counts["failed"] += 1
            continue

        oom_batch_limits: dict[int, int] = {}
        for point in points:
            context_length = int(point["context_length"])
            batch_size = int(point["batch_size"])
            oom_limit = oom_batch_limits.get(context_length)
            if oom_limit is not None and batch_size >= oom_limit:
                row = _base_row(config, variant, point, -1)
                row.update(
                    status="skipped_after_oom",
                    error_type="AdaptiveBatchLimit",
                    error=f"Skipped batch {batch_size} after OOM at batch {oom_limit}",
                )
                store.append(row)
                counts["skipped"] += repetition_count
                continue

            repetitions = range(repetition_count)
            pending_repetitions = [
                repetition
                for repetition in repetitions
                if result_identity(_base_row(config, variant, point, repetition)) not in completed
            ]
            counts["skipped"] += repetition_count - len(pending_repetitions)
            if not pending_repetitions:
                continue

            prompt = None
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

                for repetition in pending_repetitions:
                    row = _base_row(config, variant, point, repetition)
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
                previous_limit = oom_batch_limits.get(context_length)
                oom_batch_limits[context_length] = (
                    batch_size if previous_limit is None else min(previous_limit, batch_size)
                )
                if prompt is not None:
                    del prompt
                torch.cuda.empty_cache()
            except Exception as error:
                row = _base_row(config, variant, point, -1)
                row.update(status="error", error_type=type(error).__name__, error=str(error))
                store.append(row)
                counts["failed"] += 1

        del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = store.read()
    aggregates = aggregate_rows(rows)
    knees: dict[str, Any] = {}
    knee_group_fields = tuple(
        field for field in IDENTITY_FIELDS if field not in ("batch_size", "repetition")
    )
    knee_groups = {tuple(row.get(field) for field in knee_group_fields) for row in aggregates}
    for group in sorted(knee_groups, key=lambda values: tuple(map(str, values))):
        subset = [
            row
            for row in aggregates
            if tuple(row.get(field) for field in knee_group_fields) == group
        ]
        key = "|".join(
            f"{field}={value}" for field, value in zip(knee_group_fields, group)
        )
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
