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


def safe_name(values: tuple[object, ...]) -> str:
    text = "_".join(str(value) for value in values)
    return "".join(
        character if character.isalnum() or character in "-_" else "-" for character in text
    )


def identity_title(fields: list[str], values: tuple[object, ...]) -> str:
    identity = dict(zip(fields, values))
    first_line = (
        f"{str(identity['device']).upper()} {identity['dtype']} | "
        f"attention={identity['attention']} | cache={'on' if identity['use_cache'] else 'off'} | "
        f"quantization={identity['quantization']}"
    )
    hardware = str(identity["hardware_id"]).split("::", maxsplit=1)[0]
    model = Path(str(identity.get("model_id", "model"))).name
    second_line = f"{hardware} | {model}"
    if "num_threads" in identity:
        second_line += f" | threads={identity['num_threads']}"
    return f"{first_line}\n{second_line}"


def identity_slug(fields: list[str], values: tuple[object, ...]) -> str:
    identity = dict(zip(fields, values))
    hardware = str(identity["hardware_id"]).split("::", maxsplit=1)[0]
    model = Path(str(identity.get("model_id", "model"))).name
    parts = (
        hardware,
        identity["device"],
        identity["dtype"],
        identity["attention"],
        f"cache-{'on' if identity['use_cache'] else 'off'}",
        identity["quantization"],
        model,
    )
    return safe_name(parts)


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

    optional_fields = [
        "model_id",
        "model_revision",
        "compile",
        "num_threads",
        "num_interop_threads",
    ]
    group_fields = [
        "hardware_id",
        "device",
        "dtype",
        "attention",
        "use_cache",
        "quantization",
        *[field for field in optional_fields if field in data.columns],
    ]
    for identity, subset in data.groupby(group_fields, dropna=False):
        pivot = subset.pivot_table(
            index="context_length", columns="batch_size", values=args.metric, aggfunc="median"
        )
        figure_width = 12 if len(pivot.columns) > 6 else 10
        annotation_size = 10 if len(pivot.columns) > 6 else 12
        number_format = ".0f" if args.metric == "tps" else ".3g"
        fig, axis = plt.subplots(figsize=(figure_width, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=number_format,
            annot_kws={"fontsize": annotation_size},
            cmap="viridis",
            ax=axis,
        )
        axis.set_title(identity_title(group_fields, identity), fontsize=14)
        axis.set_xlabel("Batch size")
        axis.set_ylabel("Context length (tokens)")
        figure_name = identity_slug(group_fields, identity)
        fig.tight_layout()
        fig.savefig(output_dir / f"{args.metric}_heatmap_{figure_name}.png", dpi=200)
        plt.close(fig)

    frontier_groups = list(data.groupby(group_fields, dropna=False))
    for identity, subset in frontier_groups:
        summary = subset.groupby(["context_length", "batch_size"], as_index=False)[
            ["tpot_seconds", "tps"]
        ].median()
        fig, axis = plt.subplots(figsize=(9, 6))
        sns.lineplot(
            data=summary,
            x="tpot_seconds",
            y="tps",
            hue="context_length",
            style="context_length",
            markers=True,
            ax=axis,
        )
        axis.set_xlabel(LABELS["tpot_seconds"])
        axis.set_ylabel(LABELS["tps"])
        axis.set_title(
            "TPOT-throughput frontier\n" + identity_title(group_fields, identity),
            fontsize=13,
        )
        fig.tight_layout()
        suffix = f"_{identity_slug(group_fields, identity)}"
        fig.savefig(output_dir / f"tpot_throughput_frontier{suffix}.png", dpi=200)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
