"""
Create paper-style plots for Random_structures.csv.

The paper plots how a measure grows with sentence length for real trees versus
random baseline trees. This script produces the same kind of visualization for
the random-structures baseline currently available in this repository.

Outputs:
  figures/random_structures_dependency_length_by_sentence_length.png
  figures/random_structures_intervener_complexity_by_sentence_length.png
  figures/random_structures_sentence_level_stats.csv

Usage:
  python plot_random_structures_paper_style.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT = Path("Random_structures.csv")
OUT_DIR = Path("figures")
OUT_STATS = OUT_DIR / "random_structures_sentence_level_stats.csv"

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


def read_sentence_level_rows() -> list[dict[str, float | int | str]]:
    """Read edge-level rows and average measures within each sentence."""
    if not INPUT.exists():
        raise SystemExit(f"{INPUT} not found. Run construct_output_random_structures.py first.")

    by_sentence: dict[tuple[str, str, int], dict[str, object]] = {}

    with INPUT.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if len(parts) != len(COLUMNS):
                continue

            row = dict(zip(COLUMNS, parts))
            key = (row["lang"], row["kind"], int(row["sent_id"]))
            entry = by_sentence.setdefault(
                key,
                {
                    "lang": row["lang"],
                    "kind": row["kind"],
                    "sent_id": int(row["sent_id"]),
                    "sent_len": int(row["sent_len"]),
                    "dep_distances": [],
                    "dep_depths": [],
                },
            )
            entry["dep_distances"].append(float(row["dep_distance"]))  # type: ignore[index]
            entry["dep_depths"].append(float(row["dep_depth"]))  # type: ignore[index]

    sentence_rows: list[dict[str, float | int | str]] = []
    for entry in by_sentence.values():
        dep_distances = entry["dep_distances"]  # type: ignore[assignment]
        dep_depths = entry["dep_depths"]  # type: ignore[assignment]
        sentence_rows.append(
            {
                "lang": entry["lang"],
                "kind": entry["kind"],
                "sent_id": entry["sent_id"],
                "sent_len": entry["sent_len"],
                "mean_dep_distance": sum(dep_distances) / len(dep_distances),
                "mean_intervener_complexity": sum(dep_depths) / len(dep_depths),
            }
        )

    return sentence_rows


def summarize_by_length(
    sentence_rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    by_len_kind: dict[tuple[int, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in sentence_rows:
        by_len_kind[(int(row["sent_len"]), str(row["kind"]))].append(row)

    summary_rows: list[dict[str, float | int | str]] = []
    for (sent_len, kind), rows in sorted(by_len_kind.items()):
        summary_rows.append(
            {
                "sent_len": sent_len,
                "kind": kind,
                "num_sents": len(rows),
                "mean_dep_distance": mean(float(r["mean_dep_distance"]) for r in rows),
                "mean_intervener_complexity": mean(
                    float(r["mean_intervener_complexity"]) for r in rows
                ),
            }
        )
    return summary_rows


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def write_summary(summary_rows: list[dict[str, float | int | str]]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    with OUT_STATS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sent_len",
                "kind",
                "num_sents",
                "mean_dep_distance",
                "mean_intervener_complexity",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_measure(
    summary_rows: list[dict[str, float | int | str]],
    *,
    measure: str,
    ylabel: str,
    title: str,
    outfile: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.5), dpi=180)

    for kind in ("real", "random"):
        rows = [r for r in summary_rows if r["kind"] == kind]
        rows.sort(key=lambda r: int(r["sent_len"]))
        xs = [int(r["sent_len"]) for r in rows]
        ys = [float(r[measure]) for r in rows]

        ax.plot(
            xs,
            ys,
            color=COLORS[kind],
            linewidth=2.2,
            marker="o",
            markersize=4,
            label=LABELS[kind],
        )

    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Sentence length", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(left=1.8)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sentence_rows = read_sentence_level_rows()
    summary_rows = summarize_by_length(sentence_rows)
    write_summary(summary_rows)

    plot_measure(
        summary_rows,
        measure="mean_dep_distance",
        ylabel="Mean dependency length",
        title="Dependency length growth: real vs random structures",
        outfile=OUT_DIR / "random_structures_dependency_length_by_sentence_length.png",
    )
    plot_measure(
        summary_rows,
        measure="mean_intervener_complexity",
        ylabel="Mean intervener complexity",
        title="Intervener complexity growth: real vs random structures",
        outfile=OUT_DIR / "random_structures_intervener_complexity_by_sentence_length.png",
    )

    print(f"Wrote {OUT_STATS}")
    print(f"Wrote {OUT_DIR / 'random_structures_dependency_length_by_sentence_length.png'}")
    print(f"Wrote {OUT_DIR / 'random_structures_intervener_complexity_by_sentence_length.png'}")


if __name__ == "__main__":
    main()
