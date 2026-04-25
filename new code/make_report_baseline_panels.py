"""
Create report-ready multi-language panel figures for each random baseline.

This script reads the already-generated sentence-length summary CSVs from:
  - figures/by_language
  - figures_random/by_language
  - figures_pure_random/by_language

It does not modify any existing files. Instead, it writes new combined panel
figures to:
  report_figures/
    matched_random/
    pure_random_pruefer/
    pure_random_root0/

For each baseline, it creates one figure per measure, with one subplot per
language and separate lines for real and random trees.

Usage from repo root:
  python "new code\\make_report_baseline_panels.py"
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report_figures"

BASELINES = [
    {
        "key": "matched_random",
        "label": "Matched random structures",
        "input_dir": ROOT / "figures" / "by_language",
        "languages": ["ar", "es", "fi", "hi", "ja", "tr", "zh"],
        "measures": [
            ("mean_dep_distance", "dependency_length", "Mean dependency length"),
            ("mean_intervener_complexity", "intervener_complexity", "Mean intervener complexity"),
            ("mean_tree_density", "tree_density", "Mean tree density"),
            ("mean_max_arity", "max_arity", "Mean maximum arity"),
            (
                "mean_high_arity_node_proportion",
                "high_arity_nodes_ge3",
                "Proportion of nodes with arity >= 3",
            ),
        ],
    },
    {
        "key": "pure_random_pruefer",
        "label": "Pure random trees (Pruefer)",
        "input_dir": ROOT / "figures_random" / "by_language",
        "languages": ["ar", "en", "es", "fi", "hi", "ja", "tr", "zh"],
        "measures": [
            ("mean_dep_distance", "dependency_length", "Mean dependency length"),
            ("mean_intervener_complexity", "intervener_complexity", "Mean intervener complexity"),
            ("mean_tree_depth", "tree_depth", "Mean tree depth"),
            ("mean_tree_density", "tree_density", "Mean tree density"),
            ("mean_max_arity", "max_arity", "Mean maximum arity"),
            (
                "mean_high_arity_node_proportion",
                "high_arity_nodes_ge3",
                "Proportion of nodes with arity >= 3",
            ),
        ],
    },
    {
        "key": "pure_random_root0",
        "label": "Pure random trees (root 0)",
        "input_dir": ROOT / "figures_pure_random" / "by_language",
        "languages": ["ar", "en", "es", "fi", "hi", "ja", "tr", "zh"],
        "measures": [
            ("mean_dep_distance", "dependency_length", "Mean dependency length"),
            ("mean_intervener_complexity", "intervener_complexity", "Mean intervener complexity"),
            ("mean_tree_depth", "tree_depth", "Mean tree depth"),
            ("mean_tree_density", "tree_density", "Mean tree density"),
            ("mean_max_arity", "max_arity", "Mean maximum arity"),
            (
                "mean_high_arity_node_proportion",
                "high_arity_nodes_ge3",
                "Proportion of nodes with arity >= 3",
            ),
        ],
    },
]

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

COLORS = {
    "real": "#2f73b7",
    "random": "#c84c4c",
}

LABELS = {
    "real": "Real trees",
    "random": "Random trees",
}


def read_language_stats(csv_path: Path) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
    return rows


def load_baseline_rows(input_dir: Path, languages: list[str]) -> dict[str, list[dict[str, str | int | float]]]:
    data: dict[str, list[dict[str, str | int | float]]] = {}
    for lang in languages:
        csv_path = input_dir / f"{lang}_sentence_level_stats.csv"
        if csv_path.exists():
            data[lang] = read_language_stats(csv_path)
    return data


def grouped_by_kind(rows: list[dict[str, str | int | float]]) -> dict[str, list[dict[str, str | int | float]]]:
    grouped: dict[str, list[dict[str, str | int | float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["kind"])].append(row)
    for kind_rows in grouped.values():
        kind_rows.sort(key=lambda row: int(row["sent_len"]))
    return grouped


def subplot_shape(num_languages: int) -> tuple[int, int]:
    if num_languages <= 4:
        return 2, 2
    if num_languages <= 6:
        return 2, 3
    return 2, 4


def plot_baseline_measure(
    *,
    baseline_key: str,
    baseline_label: str,
    measure_key: str,
    measure_slug: str,
    measure_label: str,
    baseline_rows: dict[str, list[dict[str, str | int | float]]],
) -> Path | None:
    langs = [lang for lang in sorted(baseline_rows) if baseline_rows[lang]]
    if not langs:
        return None

    rows_n, cols_n = subplot_shape(len(langs))
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(4.8 * cols_n, 3.7 * rows_n), dpi=180)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    x_values_all: list[int] = []
    y_values_all: list[float] = []

    for ax, lang in zip(axes_list, langs):
        rows = baseline_rows[lang]
        by_kind = grouped_by_kind(rows)

        for kind in ("real", "random"):
            kind_rows = by_kind.get(kind, [])
            if not kind_rows:
                continue
            xs = [int(row["sent_len"]) for row in kind_rows]
            ys = [float(row[measure_key]) for row in kind_rows]
            x_values_all.extend(xs)
            y_values_all.extend(ys)
            ax.plot(
                xs,
                ys,
                color=COLORS[kind],
                marker="o",
                markersize=3.8,
                linewidth=2.0,
                label=LABELS[kind],
            )

        ax.set_title(f"{LANGUAGE_NAMES.get(lang, lang)} ({lang})", fontsize=11, pad=8)
        ax.set_xlabel("Sentence length", fontsize=10)
        ax.set_ylabel(measure_label, fontsize=10)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_list[len(langs):]:
        ax.axis("off")

    if x_values_all:
        x_min = min(x_values_all)
        x_max = max(x_values_all)
        for ax in axes_list[: len(langs)]:
            ax.set_xlim(x_min, x_max)

    if y_values_all:
        y_min = min(y_values_all)
        y_max = max(y_values_all)
        pad = (y_max - y_min) * 0.08 if y_max > y_min else max(0.1, y_max * 0.08 + 0.1)
        for ax in axes_list[: len(langs)]:
            ax.set_ylim(y_min - pad, y_max + pad)

    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.955),
        )

    # Keep a generous top margin so figure titles remain visible when inserted into reports.
    fig.suptitle(f"{baseline_label}: {measure_label} by sentence length", fontsize=14, y=0.992)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    baseline_out_dir = OUT_DIR / baseline_key
    baseline_out_dir.mkdir(parents=True, exist_ok=True)
    outfile = baseline_out_dir / f"{baseline_key}_{measure_slug}_panel.png"
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def main() -> None:
    written: list[Path] = []
    for baseline in BASELINES:
        baseline_rows = load_baseline_rows(baseline["input_dir"], baseline["languages"])
        for measure_key, measure_slug, measure_label in baseline["measures"]:
            outfile = plot_baseline_measure(
                baseline_key=baseline["key"],
                baseline_label=baseline["label"],
                measure_key=measure_key,
                measure_slug=measure_slug,
                measure_label=measure_label,
                baseline_rows=baseline_rows,
            )
            if outfile is not None:
                written.append(outfile)

    if written:
        print(f"Wrote {len(written)} report figures under {OUT_DIR}")
        for path in written:
            print(path)
    else:
        print("No report figures were written. Make sure the per-language summary CSVs exist first.")


if __name__ == "__main__":
    main()
