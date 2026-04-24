"""
Compare tree arity for real trees and random-structure baseline trees.

`Random_structures.csv` is edge-level, but max_arity and avg_arity are
tree-level values repeated on every edge. This script first collapses the file
to one row per (language, kind, sentence id), then plots arity versus sentence
length.

Outputs:
  figures/random_structures_max_arity_line_by_sentence_length.png
  figures/random_structures_max_arity_histogram_by_sentence_length.png
  figures/random_structures_arity_by_sentence.csv
  figures/random_structures_arity_distribution_by_length.csv

Usage:
  python plot_arity_distribution.py
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

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


def read_sentence_rows() -> list[dict[str, int | float | str]]:
    if not INPUT.exists():
        raise SystemExit(f"{INPUT} not found. Run construct_output_random_structures.py first.")

    by_sentence: dict[tuple[str, str, int], dict[str, int | float | str]] = {}

    with INPUT.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if len(parts) != len(COLUMNS):
                continue

            row = dict(zip(COLUMNS, parts))
            key = (row["lang"], row["kind"], int(row["sent_id"]))
            by_sentence.setdefault(
                key,
                {
                    "lang": row["lang"],
                    "kind": row["kind"],
                    "sent_id": int(row["sent_id"]),
                    "sent_len": int(row["sent_len"]),
                    "max_arity": int(row["max_arity"]),
                    "avg_arity": float(row["avg_arity"]),
                },
            )

    return list(by_sentence.values())


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize_by_length(
    rows: list[dict[str, int | float | str]],
) -> list[dict[str, int | float | str]]:
    grouped: dict[tuple[int, str], list[dict[str, int | float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["sent_len"]), str(row["kind"]))].append(row)

    summary_rows: list[dict[str, int | float | str]] = []
    for (sent_len, kind), group in sorted(grouped.items()):
        max_arities = [float(row["max_arity"]) for row in group]
        avg_arities = [float(row["avg_arity"]) for row in group]
        summary_rows.append(
            {
                "sent_len": sent_len,
                "kind": kind,
                "num_trees": len(group),
                "mean_max_arity": mean(max_arities),
                "median_max_arity": median(max_arities),
                "mean_avg_arity": mean(avg_arities),
            }
        )
    return summary_rows


def arity_distribution_by_length(
    rows: list[dict[str, int | float | str]],
) -> list[dict[str, int | float | str]]:
    counts = Counter(
        (int(row["sent_len"]), str(row["kind"]), int(row["max_arity"])) for row in rows
    )
    totals = Counter((int(row["sent_len"]), str(row["kind"])) for row in rows)

    dist_rows: list[dict[str, int | float | str]] = []
    for (sent_len, kind, max_arity), count in sorted(counts.items()):
        total = totals[(sent_len, kind)]
        dist_rows.append(
            {
                "sent_len": sent_len,
                "kind": kind,
                "max_arity": max_arity,
                "num_trees": count,
                "proportion_within_length": count / total,
            }
        )
    return dist_rows


def write_csvs(
    summary_rows: list[dict[str, int | float | str]],
    dist_rows: list[dict[str, int | float | str]],
) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    with (OUT_DIR / "random_structures_arity_by_sentence.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sent_len",
                "kind",
                "num_trees",
                "mean_max_arity",
                "median_max_arity",
                "mean_avg_arity",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with (OUT_DIR / "random_structures_arity_distribution_by_length.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sent_len",
                "kind",
                "max_arity",
                "num_trees",
                "proportion_within_length",
            ],
        )
        writer.writeheader()
        writer.writerows(dist_rows)


def save_figure(fig: plt.Figure, outfile: Path, *, tight_layout: bool = True) -> None:
    if tight_layout:
        fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def plot_line(summary_rows: list[dict[str, int | float | str]]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=180)

    for kind in ("real", "random"):
        rows = [row for row in summary_rows if row["kind"] == kind]
        rows.sort(key=lambda row: int(row["sent_len"]))
        ax.plot(
            [int(row["sent_len"]) for row in rows],
            [float(row["mean_max_arity"]) for row in rows],
            color=COLORS[kind],
            marker="o",
            markersize=4,
            linewidth=2.2,
            label=LABELS[kind],
        )

    ax.set_title("Tree arity growth: real vs random structures", fontsize=12, pad=12)
    ax.set_xlabel("Sentence length", fontsize=11)
    ax.set_ylabel("Mean maximum arity", fontsize=11)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, OUT_DIR / "random_structures_max_arity_line_by_sentence_length.png")


def plot_histogram_heatmap(dist_rows: list[dict[str, int | float | str]]) -> None:
    sent_lengths = sorted({int(row["sent_len"]) for row in dist_rows})
    max_arities = sorted({int(row["max_arity"]) for row in dist_rows})
    x_index = {sent_len: i for i, sent_len in enumerate(sent_lengths)}
    y_index = {arity: i for i, arity in enumerate(max_arities)}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), dpi=180, sharey=True)

    image = None
    for ax, kind in zip(axes, ("real", "random")):
        matrix = [[0.0 for _ in sent_lengths] for _ in max_arities]
        for row in dist_rows:
            if row["kind"] != kind:
                continue
            x = x_index[int(row["sent_len"])]
            y = y_index[int(row["max_arity"])]
            matrix[y][x] = float(row["proportion_within_length"])

        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(LABELS[kind], fontsize=11)
        ax.set_xlabel("Sentence length", fontsize=10)
        ax.set_xticks(range(len(sent_lengths)))
        ax.set_xticklabels(sent_lengths)
        ax.set_yticks(range(len(max_arities)))
        ax.set_yticklabels(max_arities)

    axes[0].set_ylabel("Maximum arity", fontsize=10)
    fig.suptitle("Distribution of maximum arity by sentence length", fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.13, top=0.82, wspace=0.08)
    if image is not None:
        cbar_ax = fig.add_axes([0.87, 0.15, 0.018, 0.67])
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.set_label("Proportion of trees", fontsize=10)

    save_figure(
        fig,
        OUT_DIR / "random_structures_max_arity_histogram_by_sentence_length.png",
        tight_layout=False,
    )


def main() -> None:
    sentence_rows = read_sentence_rows()
    summary_rows = summarize_by_length(sentence_rows)
    dist_rows = arity_distribution_by_length(sentence_rows)

    write_csvs(summary_rows, dist_rows)
    plot_line(summary_rows)
    plot_histogram_heatmap(dist_rows)

    print(f"Read {len(sentence_rows)} sentence-level trees from {INPUT}")
    print(f"Wrote {OUT_DIR / 'random_structures_arity_by_sentence.csv'}")
    print(f"Wrote {OUT_DIR / 'random_structures_arity_distribution_by_length.csv'}")
    print(f"Wrote {OUT_DIR / 'random_structures_max_arity_line_by_sentence_length.png'}")
    print(f"Wrote {OUT_DIR / 'random_structures_max_arity_histogram_by_sentence_length.png'}")


if __name__ == "__main__":
    main()
