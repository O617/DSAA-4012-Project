"""Deterministic, exact-token-length benchmark inputs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


DEFAULT_PROMPT = (
    "Explain one practical trade-off in machine-learning inference systems. "
    "Use precise technical language and a short concrete example. "
)


@dataclass
class PromptBatch:
    input_ids: object
    attention_mask: object
    prompt_hash: str


def exact_length_prompt(tokenizer: object, context_length: int, batch_size: int, device: str) -> PromptBatch:
    import torch

    if context_length < 1 or batch_size < 1:
        raise ValueError("context_length and batch_size must be positive")

    message = [{"role": "user", "content": DEFAULT_PROMPT * max(2, context_length // 12)}]
    if hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )[0]
    else:
        ids = tokenizer(DEFAULT_PROMPT * max(2, context_length // 12), return_tensors="pt")[
            "input_ids"
        ][0]

    if ids.numel() < context_length:
        repeats = (context_length + ids.numel() - 1) // ids.numel()
        ids = ids.repeat(repeats)
    elif ids.numel() > context_length:
        # Preserve both the chat-template prefix and generation-prompt suffix.
        prefix_length = context_length // 2
        ids = torch.cat((ids[:prefix_length], ids[-(context_length - prefix_length) :]))
    ids = ids[:context_length].contiguous()
    input_ids = ids.unsqueeze(0).repeat(batch_size, 1).to(device)
    attention_mask = torch.ones_like(input_ids)
    digest = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
    return PromptBatch(input_ids=input_ids, attention_mask=attention_mask, prompt_hash=digest)
