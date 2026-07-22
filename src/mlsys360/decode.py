"""Manual fixed-work autoregressive decoding with token-level timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memory import RSSMonitor, collect_memory, reset_cuda_peaks
from .timing import timestamp


@dataclass
class DecodeResult:
    ttft_seconds: float
    tpot_seconds: float
    decode_seconds: float
    end_to_end_seconds: float
    tps: float
    end_to_end_tps: float
    token_timestamps: list[float]
    token_intervals: list[float]
    generated_token_ids: list[list[int]]
    prefill_logits_shape: list[int]
    max_logits_positions: int
    memory: dict[str, int | None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ttft_seconds": self.ttft_seconds,
            "tpot_seconds": self.tpot_seconds,
            "decode_seconds": self.decode_seconds,
            "end_to_end_seconds": self.end_to_end_seconds,
            "tps": self.tps,
            "end_to_end_tps": self.end_to_end_tps,
            "token_timestamps": self.token_timestamps,
            "token_intervals": self.token_intervals,
            "generated_token_ids": self.generated_token_ids,
            "prefill_logits_shape": self.prefill_logits_shape,
            "max_logits_positions": self.max_logits_positions,
            **self.memory,
        }


def _next_token(logits: Any) -> Any:
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def _forward_last_logits(model: Any, **kwargs: Any) -> Any:
    """Run one model forward while materializing only the final logit position."""
    try:
        return model(**kwargs, logits_to_keep=1)
    except TypeError as error:
        if "logits_to_keep" in str(error):
            raise RuntimeError(
                "The benchmark model must support logits_to_keep=1; add a "
                "model-specific last-logit adapter before benchmarking this architecture"
            ) from error
        raise


def _logits_shape(logits: Any, batch_size: int) -> list[int]:
    shape = [int(value) for value in logits.shape]
    if len(shape) != 3 or shape[0] != batch_size or shape[1] != 1:
        raise RuntimeError(
            "Benchmark forwards must return [batch, 1, vocabulary] logits; "
            f"received {shape}"
        )
    return shape


def fixed_work_decode(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    output_tokens: int,
    device: str,
    use_cache: bool = True,
) -> DecodeResult:
    """Generate exactly ``output_tokens`` with greedy decoding.

    The first forward pass consumes the complete prompt. Every timestamp is taken
    after device synchronization, so CUDA results include actual execution rather
    than asynchronous launch time.
    """
    import torch

    if output_tokens < 2:
        raise ValueError("output_tokens must be at least 2")

    batch_size = int(input_ids.shape[0])
    generated = []
    token_times: list[float] = []
    reset_cuda_peaks(device)

    with RSSMonitor() as rss_monitor, torch.inference_mode():
        start = timestamp(device)
        outputs = _forward_last_logits(
            model, input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache
        )
        prefill_logits_shape = _logits_shape(outputs.logits, batch_size)
        next_token = _next_token(outputs.logits)
        generated.append(next_token)
        token_times.append(timestamp(device))

        full_ids = torch.cat((input_ids, next_token), dim=1)
        full_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=1)
        past_key_values = outputs.past_key_values if use_cache else None

        for _ in range(output_tokens - 1):
            if use_cache:
                outputs = _forward_last_logits(
                    model,
                    input_ids=next_token,
                    attention_mask=full_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
            else:
                outputs = _forward_last_logits(
                    model, input_ids=full_ids, attention_mask=full_mask, use_cache=False
                )
            _logits_shape(outputs.logits, batch_size)
            next_token = _next_token(outputs.logits)
            generated.append(next_token)
            token_times.append(timestamp(device))
            full_ids = torch.cat((full_ids, next_token), dim=1)
            full_mask = torch.cat((full_mask, torch.ones_like(next_token)), dim=1)

        end = token_times[-1]

    ttft = token_times[0] - start
    decode_seconds = token_times[-1] - token_times[0]
    intervals = [right - left for left, right in zip(token_times, token_times[1:])]
    generated_ids = torch.cat(generated, dim=1).detach().cpu().tolist()
    memory = collect_memory(device, rss_monitor.peak_bytes).as_dict()
    return DecodeResult(
        ttft_seconds=ttft,
        tpot_seconds=decode_seconds / (output_tokens - 1),
        decode_seconds=decode_seconds,
        end_to_end_seconds=end - start,
        tps=batch_size * (output_tokens - 1) / decode_seconds,
        end_to_end_tps=batch_size * output_tokens / (end - start),
        token_timestamps=[value - start for value in token_times],
        token_intervals=intervals,
        generated_token_ids=generated_ids,
        prefill_logits_shape=prefill_logits_shape,
        max_logits_positions=1,
        memory=memory,
    )
