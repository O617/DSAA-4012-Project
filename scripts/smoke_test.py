#!/usr/bin/env python3
import argparse
import json

from mlsys360.config import apply_overrides, load_config
from mlsys360.decode import fixed_work_decode
from mlsys360.model import load_model
from mlsys360.prompts import exact_length_prompt


EXPECTED = {
    "num_hidden_layers": 32,
    "hidden_size": 960,
    "intermediate_size": 2560,
    "num_attention_heads": 15,
    "num_key_value_heads": 5,
    "max_position_embeddings": 8192,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    bundle = load_model(config)
    prompt = exact_length_prompt(bundle.tokenizer, 32, 1, bundle.device)
    result = fixed_work_decode(
        bundle.model, prompt.input_ids, prompt.attention_mask, 8, bundle.device, True
    )
    mismatches = {
        key: {"expected": value, "actual": bundle.metadata.get(key)}
        for key, value in EXPECTED.items()
        if bundle.metadata.get(key) != value
    }
    report = {
        "status": "ok" if not mismatches else "model_metadata_mismatch",
        "model": bundle.metadata,
        "mismatches": mismatches,
        "prompt_hash": prompt.prompt_hash,
        "metrics": result.as_dict(),
    }
    print(json.dumps(report, indent=2))
    return 0 if not mismatches else 2


if __name__ == "__main__":
    raise SystemExit(main())

