"""Perplexity and lm-evaluation-harness quality measurements."""

from __future__ import annotations

import math
from typing import Any

from .model import load_model


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
    previous_end = 0
    for begin in range(0, tokens.size(1), stride):
        end = min(begin + max_length, tokens.size(1))
        target_length = end - previous_end
        input_ids = tokens[:, begin:end]
        labels = input_ids.clone()
        labels[:, :-target_length] = -100
        with torch.inference_mode():
            output = bundle.model(input_ids, labels=labels)
        predicted = max(target_length - 1, 0)
        negative_log_likelihoods.append(float(output.loss) * predicted)
        token_count += predicted
        previous_end = end
        if end == tokens.size(1):
            break
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
