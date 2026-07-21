#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


LABELS = {
    "ttft_seconds": "TTFT (seconds)",
    "tpot_seconds": "TPOT (seconds/token)",
    "tps": "Decode throughput (tokens/second)",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", nargs="+", help="One or more benchmark JSONL files")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--metric", choices=tuple(LABELS), default="tps")
    args = parser.parse_args()
    frames = [pd.read_json(path, lines=True) for path in args.raw]
    data = pd.concat(frames, ignore_index=True)
    data = data[data["status"] == "ok"].copy()
    if data.empty:
        raise SystemExit("No successful result rows found")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    group_fields = ["device", "attention", "use_cache", "quantization"]
    for identity, subset in data.groupby(group_fields, dropna=False):
        pivot = subset.pivot_table(
            index="context_length", columns="batch_size", values=args.metric, aggfunc="median"
        )
        fig, axis = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".3g", cmap="viridis", ax=axis)
        axis.set_title(" | ".join(map(str, identity)))
        axis.set_xlabel("Batch size")
        axis.set_ylabel("Context length (tokens)")
        figure_name = "_".join(str(value).replace("/", "-") for value in identity)
        fig.tight_layout()
        fig.savefig(output_dir / f"{args.metric}_heatmap_{figure_name}.png", dpi=200)
        plt.close(fig)

    summary = data.groupby(["device", "context_length", "batch_size"], as_index=False)[
        ["tpot_seconds", "tps"]
    ].median()
    fig, axis = plt.subplots(figsize=(9, 6))
    sns.lineplot(
        data=summary,
        x="tpot_seconds",
        y="tps",
        hue="device",
        style="context_length",
        markers=True,
        ax=axis,
    )
    axis.set_xlabel(LABELS["tpot_seconds"])
    axis.set_ylabel(LABELS["tps"])
    axis.set_title("TPOT-throughput frontier")
    fig.tight_layout()
    fig.savefig(output_dir / "tpot_throughput_frontier.png", dpi=200)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
