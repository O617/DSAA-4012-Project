"""Device-aware quantization with explicit backend reporting."""

from __future__ import annotations

from typing import Any


def quantize_model(model: Any, mode: str, device: str) -> tuple[Any, dict[str, Any]]:
    if mode in ("none", "", None):
        return model, {"mode": "none", "backend": "native", "modules": []}

    if mode == "dynamic_w8a8" and str(device) == "cpu":
        import torch

        quantized = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
        )
        modules = [name for name, module in quantized.named_modules() if "quantized" in type(module).__module__]
        return quantized, {
            "mode": mode,
            "backend": "torch_ao_dynamic_cpu",
            "modules": modules,
            "weight_granularity": "per-channel when supported",
            "activation_granularity": "dynamic per-tensor",
        }

    try:
        from torchao.quantization import quantize_
        from torchao.quantization.quant_api import (
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
        )
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            f"{mode} on {device} requires a compatible torchao installation; "
            "install the 'quantization' extra and record the exact version"
        ) from error

    if mode == "dynamic_w8a8":
        recipe = Int8DynamicActivationInt8WeightConfig()
    elif mode == "weight_only_w8a16":
        recipe = Int8WeightOnlyConfig()
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")

    quantize_(model, recipe)
    modules = [name for name, module in model.named_modules() if "AffineQuantized" in type(module).__name__]
    return model, {"mode": mode, "backend": "torchao", "modules": modules}

