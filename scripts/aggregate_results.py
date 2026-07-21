#!/usr/bin/env python3
import argparse

from mlsys360.experiments import save_aggregates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("--output", default="results/processed/aggregates.csv")
    args = parser.parse_args()
    count = save_aggregates(args.raw, args.output)
    print(f"Wrote {count} aggregate rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

