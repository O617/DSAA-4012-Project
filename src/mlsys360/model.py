"""Model/tokenizer loading and checkpoint metadata validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .quantization import quantize_model


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    device: str
    dtype: str
    quantization: dict[str, Any]
    metadata: dict[str, Any]


def resolve_dtype(name: str) -> Any:
    import torch

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {name}") from error


def local_model_artifact_hashes(model_id: str) -> list[dict[str, Any]]:
    """Hash local inference artifacts required to identify an exact model snapshot."""
    root = Path(model_id)
    if not root.is_dir():
        return []
    exact_names = {
        "config.json",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    artifacts = []
    for path in sorted(candidate for candidate in root.iterdir() if candidate.is_file()):
        is_weight = (
            (path.name.startswith("model") and path.suffix == ".safetensors")
            or (path.name.startswith("pytorch_model") and path.suffix == ".bin")
            or path.name in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
        )
        if path.name not in exact_names and not is_weight:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        artifacts.append(
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        )
    return artifacts


def load_model(config: dict[str, Any], quantization: str = "none") -> ModelBundle:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = config["model"]
    runtime = config["runtime"]
    device = str(runtime["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    dtype = resolve_dtype(str(runtime.get("dtype", "float32")))
    attention = str(runtime.get("attention", "sdpa"))
    requested_revision = model_cfg.get("revision", "main")
    common = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }
    if requested_revision is not None:
        common["revision"] = requested_revision
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"], **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        dtype=dtype,
        attn_implementation=attention,
        **common,
    )
    model.eval().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model, quant_metadata = quantize_model(model, quantization, device)
    if bool(runtime.get("compile", False)):
        model = torch.compile(model, mode="reduce-overhead")

    cfg = model.config
    metadata = {
        "model_id": model_cfg["model_id"],
        "model_source": "local" if Path(model_cfg["model_id"]).exists() else "huggingface",
        "requested_revision": requested_revision,
        "resolved_revision": getattr(cfg, "_commit_hash", None),
        "architecture": list(getattr(cfg, "architectures", []) or []),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "parameter_count": parameter_count,
        "requested_attention": attention,
        "resolved_attention": getattr(cfg, "_attn_implementation", None),
        "local_artifacts": local_model_artifact_hashes(str(model_cfg["model_id"])),
    }
    return ModelBundle(model, tokenizer, device, str(runtime.get("dtype")), quant_metadata, metadata)
