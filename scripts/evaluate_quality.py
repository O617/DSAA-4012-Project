#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from mlsys360.config import apply_overrides, load_config
from mlsys360.quality import evaluate_perplexity, evaluate_tasks


def evaluation_batch_size(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "batch size must be a positive integer or 'auto'"
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("batch size must be a positive integer or 'auto'")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--metric", choices=("perplexity", "tasks"), default="perplexity")
    parser.add_argument("--tasks", nargs="+", default=["hellaswag", "arc_easy"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--eval-batch-size", type=evaluation_batch_size, default=1)
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument(
        "--text-file",
        help="Local text file for offline perplexity evaluation; preserves dataset labels",
    )
    parser.add_argument("--output", default="results/raw/quality.jsonl")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    if args.metric == "perplexity":
        result = evaluate_perplexity(
            config,
            args.quantization,
            max_samples=args.max_samples,
            text_file=args.text_file,
        )
    else:
        result = evaluate_tasks(
            config,
            args.quantization,
            args.tasks,
            args.limit,
            batch_size=args.eval_batch_size,
            log_samples=args.log_samples,
        )
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
