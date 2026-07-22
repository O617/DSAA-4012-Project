"""Configuration loading, inheritance, and command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> Dict[str, Any]:
    return _load_config(Path(path).resolve(), ())


def _load_config(path: Path, parents: tuple[Path, ...]) -> Dict[str, Any]:
    import yaml

    if path in parents:
        chain = " -> ".join(str(item) for item in (*parents, path))
        raise ValueError(f"Circular configuration inheritance: {chain}")
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    if not isinstance(current, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    parent = current.pop("extends", None)
    if not parent:
        return current
    if not isinstance(parent, str):
        raise ValueError(f"Configuration 'extends' must be a path string: {path}")
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return deep_merge(_load_config(parent_path.resolve(), (*parents, path)), current)


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    import yaml

    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must have KEY=VALUE form: {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        if not dotted_key or any(not part for part in dotted_key.split(".")):
            raise ValueError(f"Override key must be a non-empty dotted path: {dotted_key!r}")
        value = yaml.safe_load(raw_value)
        cursor = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot set nested value below {part!r}")
        cursor[parts[-1]] = value
    return result


def validate_config(config: Dict[str, Any]) -> None:
    required = ("model", "runtime", "workload", "output")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    for section in required:
        if not isinstance(config[section], dict):
            raise ValueError(f"Configuration section {section!r} must be a mapping")

    model = config["model"]
    runtime = config["runtime"]
    workload = config["workload"]
    output = config["output"]
    if not str(model.get("model_id", "")).strip():
        raise ValueError("model.model_id must be non-empty")
    if not str(runtime.get("device", "")).strip():
        raise ValueError("runtime.device must be non-empty")
    for field in ("num_threads", "num_interop_threads"):
        value = runtime.get(field)
        if value is not None and int(value) < 1:
            raise ValueError(f"runtime.{field} must be positive")
    try:
        output_tokens = int(workload.get("output_tokens", 0))
        warmups = int(workload.get("warmups", 0))
        repetitions = int(workload.get("repetitions", 1))
    except (TypeError, ValueError) as error:
        raise ValueError("output_tokens, warmups, and repetitions must be integers") from error
    if output_tokens < 2:
        raise ValueError("workload.output_tokens must be at least 2 for TPOT")
    if warmups < 0:
        raise ValueError("workload.warmups cannot be negative")
    if repetitions < 1:
        raise ValueError("workload.repetitions must be at least 1")

    points = workload.get("points")
    if points is not None:
        if not isinstance(points, list) or not points:
            raise ValueError("workload.points must be a non-empty list")
        if any(not isinstance(point, dict) for point in points):
            raise ValueError("Every workload point must be a mapping")
        pairs = [(point.get("context_length"), point.get("batch_size")) for point in points]
    else:
        contexts = workload.get("context_lengths")
        batches = workload.get("batch_sizes")
        if not isinstance(contexts, list) or not contexts:
            raise ValueError("workload.context_lengths must be a non-empty list")
        if not isinstance(batches, list) or not batches:
            raise ValueError("workload.batch_sizes must be a non-empty list")
        pairs = [(context, batch) for context in contexts for batch in batches]
    try:
        invalid_point = any(int(context) < 1 or int(batch) < 1 for context, batch in pairs)
    except (TypeError, ValueError) as error:
        raise ValueError("Context lengths and batch sizes must be integers") from error
    if invalid_point:
        raise ValueError("All context lengths and batch sizes must be positive")

    matrix = config.get("matrix", {})
    if not isinstance(matrix, dict):
        raise ValueError("matrix must be a mapping")
    if any(not isinstance(values, list) or not values for values in matrix.values()):
        raise ValueError("Every matrix value must be a non-empty list")
    if not str(output.get("path", "")).strip():
        raise ValueError("output.path must be non-empty")
