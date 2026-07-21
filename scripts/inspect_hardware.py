#!/usr/bin/env python3
import argparse
import json

from mlsys360.hardware import write_hardware_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/hardware.json")
    args = parser.parse_args()
    print(json.dumps(write_hardware_report(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

