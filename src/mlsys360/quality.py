"""Perplexity and lm-evaluation-harness quality measurements."""

from __future__ import annotations

import math
from typing import Any

from .model import load_model


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
) -> dict[str, Any]:
    import torch
    from datasets import load_dataset

    bundle = load_model(config, quantization=quantization)
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    texts = [text for text in dataset["text"][:max_samples] if text.strip()]
    tokens = bundle.tokenizer("\n\n".join(texts), return_tensors="pt").input_ids.to(bundle.device)
    max_length = min(
        int(getattr(bundle.model.config, "max_position_embeddings", 2048)), 2048
    )
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
        "quantization": quantization,
        "model": bundle.metadata,
        "quantization_details": bundle.quantization,
    }


def evaluate_tasks(
    config: dict[str, Any], quantization: str, tasks: list[str], limit: int | None = None
) -> dict[str, Any]:
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError as error:
        raise RuntimeError("Install the 'quality' extra to run task accuracy evaluation") from error

    bundle = load_model(config, quantization=quantization)
    wrapper = HFLM(pretrained=bundle.model, tokenizer=bundle.tokenizer, device=bundle.device)
    results = evaluator.simple_evaluate(model=wrapper, tasks=tasks, limit=limit, random_seed=4012)
    return {
        "metric": "task_accuracy",
        "tasks": tasks,
        "limit": limit,
        "quantization": quantization,
        "results": results.get("results", {}),
        "samples": results.get("samples", {}),
        "model": bundle.metadata,
        "quantization_details": bundle.quantization,
    }
