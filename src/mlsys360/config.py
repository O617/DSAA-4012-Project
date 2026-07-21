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
    import yaml

    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    parent = current.pop("extends", None)
    if not parent:
        return current
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return deep_merge(load_config(parent_path), current)


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    import yaml

    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must have KEY=VALUE form: {item!r}")
        dotted_key, raw_value = item.split("=", 1)
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
    workload = config["workload"]
    if int(workload.get("output_tokens", 0)) < 2:
        raise ValueError("workload.output_tokens must be at least 2 for TPOT")
