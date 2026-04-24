"""
Plot the proportion of high-arity nodes versus sentence length.

For each real/random tree in Random_structures.csv, this reconstructs node
out-degrees from the edge list and computes:

    number of nodes with arity >= k / number of nodes in the sentence

The default threshold is k=3.

Outputs for k=3:
  figures/random_structures_high_arity_nodes_ge3_by_sentence_length.png
  figures/random_structures_high_arity_nodes_ge3_by_sentence_length.csv

Usage:
  python plot_high_arity_nodes.py
  python plot_high_arity_nodes.py --threshold 4
"""

from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT = Path("Random_structures.csv")
OUT_DIR = Path("figures")

COLUMNS = [
    "lang",
    "kind",
    "sent_id",
    "sent_len",
    "max_arity",
    "avg_arity",
    "projection_degree",
    "gap_degree",
    "k_illnestedness",
    "edge",
    "direction",
    "dep_distance",
    "dep_depth",
    "projectivity",
    "edge_degree",
    "endpoint_crossing",
    "hdd",
]

COLORS = {
    "real": "#2f73b7",
    "random": "#c84c4c",
}

LABELS = {
    "real": "Real trees",
    "random": "Random structures",
}


def sentence_nodes(kind: str, sent_len: int) -> set[int]:
    """Return the expected non-abstract nodes for this tree.

    Real SUD trees use token ids 1..N. Random trees generated here use labels
    0..N-1 plus an abstract root, which was not written to the CSV.
    """
    if kind == "random":
        return set(range(sent_len))
    return set(range(1, sent_len + 1))


def read_tree_arities() -> list[dict[str, int | float | str]]:
    if not INPUT.exists():
        raise SystemExit(f"{INPUT} not found. Run construct_output_random_structures.py first.")

    trees: dict[tuple[str, str, int], dict[str, object]] = {}

    with INPUT.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if len(parts) != len(COLUMNS):
                continue

            row = dict(zip(COLUMNS, parts))
            kind = row["kind"]
            sent_len = int(row["sent_len"])
            sent_id = int(row["sent_id"])
            key = (row["lang"], kind, sent_id)

            entry = trees.setdefault(
                key,
                {
                    "lang": row["lang"],
                    "kind": kind,
                    "sent_id": sent_id,
                    "sent_len": sent_len,
                    "nodes": sentence_nodes(kind, sent_len),
                    "outdegree": Counter(),
                },
            )

            head, dependent = ast.literal_eval(row["edge"])
            nodes = entry["nodes"]  # type: ignore[assignment]
            outdegree = entry["outdegree"]  # type: ignore[assignment]
            nodes.update([head, dependent])
            outdegree[head] += 1

    tree_rows: list[dict[str, int | float | str]] = []
    for entry in trees.values():
        nodes = entry["nodes"]  # type: ignore[assignment]
        outdegree = entry["outdegree"]  # type: ignore[assignment]
        arities = [outdegree[node] for node in nodes]
        tree_rows.append(
            {
                "lang": entry["lang"],
                "kind": entry["kind"],
                "sent_id": entry["sent_id"],
                "sent_len": entry["sent_len"],
                "num_nodes": len(nodes),
                "max_node_arity": max(arities),
            }
        )

    return tree_rows


def summarize_high_arity(
    threshold: int,
) -> tuple[list[dict[str, int | float | str]], list[dict[str, int | float | str]]]:
    if not INPUT.exists():
        raise SystemExit(f"{INPUT} not found. Run construct_output_random_structures.py first.")

    trees: dict[tuple[str, str, int], dict[str, object]] = {}

    with INPUT.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if len(parts) != len(COLUMNS):
                continue

            row = dict(zip(COLUMNS, parts))
            kind = row["kind"]
            sent_len = int(row["sent_len"])
            sent_id = int(row["sent_id"])
            key = (row["lang"], kind, sent_id)

            entry = trees.setdefault(
                key,
                {
                    "lang": row["lang"],
                    "kind": kind,
                    "sent_id": sent_id,
                    "sent_len": sent_len,
                    "nodes": sentence_nodes(kind, sent_len),
                    "outdegree": Counter(),
                },
            )

            head, dependent = ast.literal_eval(row["edge"])
            nodes = entry["nodes"]  # type: ignore[assignment]
            outdegree = entry["outdegree"]  # type: ignore[assignment]
            nodes.update([head, dependent])
            outdegree[head] += 1

    tree_rows: list[dict[str, int | float | str]] = []
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)

    for entry in trees.values():
        nodes = entry["nodes"]  # type: ignore[assignment]
        outdegree = entry["outdegree"]  # type: ignore[assignment]
        num_high_arity_nodes = sum(1 for node in nodes if outdegree[node] >= threshold)
        proportion = num_high_arity_nodes / len(nodes)

        row = {
            "sent_len": entry["sent_len"],
            "kind": entry["kind"],
            "sent_id": entry["sent_id"],
            "num_nodes": len(nodes),
            "num_high_arity_nodes": num_high_arity_nodes,
            "proportion_high_arity_nodes": proportion,
        }
        tree_rows.append(row)
        grouped[(int(entry["sent_len"]), str(entry["kind"]))].append(proportion)

    summary_rows: list[dict[str, int | float | str]] = []
    for (sent_len, kind), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "sent_len": sent_len,
                "kind": kind,
                "num_trees": len(values),
                "mean_proportion_high_arity_nodes": sum(values) / len(values),
            }
        )

    return tree_rows, summary_rows


def write_summary(threshold: int, summary_rows: list[dict[str, int | float | str]]) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    outfile = OUT_DIR / f"random_structures_high_arity_nodes_ge{threshold}_by_sentence_length.csv"
    with outfile.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sent_len",
                "kind",
                "num_trees",
                "mean_proportion_high_arity_nodes",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return outfile


def plot_summary(threshold: int, summary_rows: list[dict[str, int | float | str]]) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    outfile = OUT_DIR / f"random_structures_high_arity_nodes_ge{threshold}_by_sentence_length.png"

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=180)

    for kind in ("real", "random"):
        rows = [row for row in summary_rows if row["kind"] == kind]
        rows.sort(key=lambda row: int(row["sent_len"]))
        ax.plot(
            [int(row["sent_len"]) for row in rows],
            [float(row["mean_proportion_high_arity_nodes"]) for row in rows],
            color=COLORS[kind],
            marker="o",
            markersize=4,
            linewidth=2.2,
            label=LABELS[kind],
        )

    ax.set_title(
        f"High-arity nodes by sentence length: arity >= {threshold}",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel("Sentence length", fontsize=11)
    ax.set_ylabel(f"Proportion of nodes with arity >= {threshold}", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="High-arity cutoff k. Defaults to 3, meaning arity >= 3.",
    )
    args = parser.parse_args()

    _, summary_rows = summarize_high_arity(args.threshold)
    csv_out = write_summary(args.threshold, summary_rows)
    png_out = plot_summary(args.threshold, summary_rows)

    print(f"Wrote {csv_out}")
    print(f"Wrote {png_out}")


if __name__ == "__main__":
    main()
