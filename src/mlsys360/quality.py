"""Perplexity and lm-evaluation-harness quality measurements."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from .model import load_model
from .provenance import software_versions


def json_safe(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=str)]
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}.{name}"
    return str(value)


def file_provenance(path: str | Path) -> dict[str, Any]:
    """Hash one local evaluation artifact instead of trusting a label in YAML."""
    source = Path(path)
    content = source.read_bytes()
    return {
        "path": str(source),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def task_source_provenance(task_configs: dict[str, Any]) -> dict[str, Any]:
    """Record actual task-YAML and split-file hashes reported by lm-eval."""
    provenance: dict[str, Any] = {}
    for task_name, task_config in task_configs.items():
        task_files: dict[str, Any] = {"data_files": {}}
        metadata = task_config.get("metadata", {})
        config_source = metadata.get("config_source")
        if config_source and Path(config_source).is_file():
            task_files["task_config"] = file_provenance(config_source)
        data_files = task_config.get("dataset_kwargs", {}).get("data_files", {})
        if isinstance(data_files, str):
            data_files = {"unspecified": data_files}
        for split, filename in sorted(data_files.items()):
            if Path(filename).is_file():
                task_files["data_files"][str(split)] = file_provenance(filename)
        provenance[str(task_name)] = task_files
    return provenance


def perplexity_windows(
    sequence_length: int, max_length: int, stride: int
) -> list[tuple[int, int, int, int]]:
    """Return ``(begin, end, target_length, scored_tokens)`` windows."""
    if sequence_length < 1:
        return []
    if max_length < 2:
        raise ValueError("max_length must be at least 2 for causal scoring")
    if stride < 1 or stride > max_length:
        raise ValueError(f"stride must be between 1 and {max_length}, got {stride}")

    windows = []
    previous_end = 0
    for begin in range(0, sequence_length, stride):
        end = min(begin + max_length, sequence_length)
        target_length = end - previous_end
        scored_tokens = max(target_length - 1, 0) if begin == 0 else target_length
        windows.append((begin, end, target_length, scored_tokens))
        previous_end = end
        if end == sequence_length:
            break
    return windows


def evaluate_perplexity(
    config: dict[str, Any],
    quantization: str,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_samples: int = 200,
    stride: int = 512,
    text_file: str | Path | None = None,
) -> dict[str, Any]:
    import torch

    bundle = load_model(config, quantization=quantization)
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    source: dict[str, Any]
    if text_file is None:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, dataset_config, split=split)
        texts = [text for text in dataset["text"][:max_samples] if text.strip()]
        source = {"type": "huggingface_dataset"}
    else:
        path = Path(text_file)
        content = path.read_bytes()
        lines = content.decode("utf-8").splitlines()
        texts = [text for text in lines[:max_samples] if text.strip()]
        source = {
            "type": "local_text_file",
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "line_count": len(lines),
        }
    tokens = bundle.tokenizer("\n\n".join(texts), return_tensors="pt").input_ids.to(bundle.device)
    max_length = min(int(getattr(bundle.model.config, "max_position_embeddings", 2048)), 2048)
    negative_log_likelihoods = []
    token_count = 0
    for begin, end, target_length, predicted in perplexity_windows(
        tokens.size(1), max_length, stride
    ):
        input_ids = tokens[:, begin:end]
        labels = input_ids.clone()
        labels[:, :-target_length] = -100
        with torch.inference_mode():
            output = bundle.model(input_ids, labels=labels)
        negative_log_likelihoods.append(float(output.loss) * predicted)
        token_count += predicted
    if token_count == 0:
        raise RuntimeError("Quality dataset produced no scorable tokens")
    mean_nll = sum(negative_log_likelihoods) / token_count
    return {
        "metric": "perplexity",
        "value": math.exp(mean_nll),
        "mean_negative_log_likelihood": mean_nll,
        "evaluated_tokens": token_count,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "max_samples": max_samples,
        "source": source,
        "quantization": quantization,
        "model": bundle.metadata,
        "quantization_details": bundle.quantization,
        "software": software_versions(),
    }


def evaluate_tasks(
    config: dict[str, Any],
    quantization: str,
    tasks: list[str],
    limit: int | None = None,
    batch_size: int | str = 1,
    log_samples: bool = False,
) -> dict[str, Any]:
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError as error:
        raise RuntimeError("Install the 'quality' extra to run task accuracy evaluation") from error

    bundle = load_model(config, quantization=quantization)
    wrapper = HFLM(
        pretrained=bundle.model,
        tokenizer=bundle.tokenizer,
        device=bundle.device,
        batch_size=batch_size,
    )
    results = evaluator.simple_evaluate(
        model=wrapper,
        tasks=tasks,
        limit=limit,
        log_samples=log_samples,
        random_seed=4012,
    )
    task_configs = results.get("configs", {})
    output = {
        "metric": "task_accuracy",
        "tasks": tasks,
        "limit": limit,
        "batch_size": batch_size,
        "quantization": quantization,
        "results": results.get("results", {}),
        "task_configs": json_safe(task_configs),
        "task_sources": task_source_provenance(task_configs),
        "task_versions": results.get("versions", {}),
        "model": bundle.metadata,
        "quantization_details": bundle.quantization,
        "software": software_versions(),
    }
    if log_samples:
        output["samples"] = results.get("samples", {})
    return output
