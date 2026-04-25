"""
Create real-tree-only overlay plots for the pure-random Pruefer baseline.

This script reads the existing sentence-length summary CSVs in:
  figures_random/by_language/

It does not modify any existing files. It writes new figures to:
  report_figures/pure_random_pruefer_real_only/

Each output figure shows one measure against sentence length, with one line per
language and only the `real` rows from the Pruefer-based pure-random analysis.

Usage from the repo root:
  python "new code\\make_real_only_pruefer_overlays.py"
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "figures_random" / "by_language"
OUT_DIR = ROOT / "report_figures" / "pure_random_pruefer_real_only"

LANGUAGES = ["ar", "en", "es", "fi", "hi", "ja", "tr", "zh"]
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "hi": "Hindi",
    "ja": "Japanese",
    "tr": "Turkish",
    "zh": "Chinese",
}

MEASURES = [
    ("mean_dep_distance", "dependency_length", "Mean dependency length"),
    ("mean_intervener_complexity", "intervener_complexity", "Mean intervener complexity"),
    ("mean_tree_density", "tree_density", "Mean tree density"),
    ("mean_tree_depth", "tree_depth", "Mean tree depth"),
    ("mean_max_arity", "max_arity", "Mean maximum arity"),
]

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
]


def read_real_rows(csv_path: Path) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("kind") != "real":
                continue
            parsed: dict[str, str | int | float] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = ""
                elif key in {"sent_len", "num_trees"}:
                    parsed[key] = int(value)
                elif key in {"lang", "kind"}:
                    parsed[key] = value
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    rows.sort(key=lambda row: int(row["sent_len"]))
    return rows


def load_all_real_rows() -> dict[str, list[dict[str, str | int | float]]]:
    data: dict[str, list[dict[str, str | int | float]]] = {}
    for lang in LANGUAGES:
        csv_path = INPUT_DIR / f"{lang}_sentence_level_stats.csv"
        if csv_path.exists():
            rows = read_real_rows(csv_path)
            if rows:
                data[lang] = rows
    return data


def plot_measure(
    measure_key: str,
    measure_slug: str,
    measure_label: str,
    all_rows: dict[str, list[dict[str, str | int | float]]],
) -> Path | None:
    if not all_rows:
        return None

    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=180)

    x_values_all: list[int] = []
    y_values_all: list[float] = []

    for index, lang in enumerate(LANGUAGES):
        rows = all_rows.get(lang, [])
        if not rows:
            continue
        xs = [int(row["sent_len"]) for row in rows]
        ys = [float(row[measure_key]) for row in rows]
        x_values_all.extend(xs)
        y_values_all.extend(ys)
        ax.plot(
            xs,
            ys,
            color=COLORS[index % len(COLORS)],
            marker="o",
            markersize=4,
            linewidth=1.9,
            label=LANGUAGE_NAMES.get(lang, lang),
        )

    if not x_values_all:
        plt.close(fig)
        return None

    ax.set_xlabel("Sentence length", fontsize=12)
    ax.set_ylabel(measure_label, fontsize=12)
    ax.set_title(
        f"Real trees: {measure_label} by sentence length\n"
        f"(languages included in the Pruefer-baseline analysis)",
        fontsize=14,
        pad=12,
    )
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    x_min = min(x_values_all)
    x_max = max(x_values_all)
    ax.set_xlim(x_min, x_max)

    y_min = min(y_values_all)
    y_max = max(y_values_all)
    pad = (y_max - y_min) * 0.08 if y_max > y_min else max(0.1, y_max * 0.08 + 0.1)
    ax.set_ylim(y_min - pad, y_max + pad)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = OUT_DIR / f"pure_random_pruefer_real_{measure_slug}_overlay.png"
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def main() -> None:
    all_rows = load_all_real_rows()
    written: list[Path] = []
    for measure_key, measure_slug, measure_label in MEASURES:
        outfile = plot_measure(measure_key, measure_slug, measure_label, all_rows)
        if outfile is not None:
            written.append(outfile)

    if written:
        print(f"Wrote {len(written)} real-only overlay plots to {OUT_DIR}")
        for path in written:
            print(path)
    else:
        print("No plots were written. Make sure the figures_random/by_language CSVs exist.")


if __name__ == "__main__":
    main()
