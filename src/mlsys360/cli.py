"""Small command-line helpers shared by repository scripts."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .config import apply_overrides, load_config
from .experiments import run_experiment


def benchmark_main(default_config: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="Override a YAML value"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print merged config and exit")
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2))
    return 0
