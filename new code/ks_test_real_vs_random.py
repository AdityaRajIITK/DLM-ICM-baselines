"""
Run Kolmogorov-Smirnov tests comparing real trees with available random-tree datasets.

This script reads the existing edge-level datasets, reconstructs sentence-level
measure rows, runs two-sample KS tests for real vs random distributions, and
writes new CSV summaries and plots without modifying any existing files.

Outputs:
  ks_results/overall_ks_results.csv
  ks_results/by_language_ks_results.csv
  ks_results/sentence_level/{dataset_key}_sentence_level_stats.csv
  ks_results/plots/{dataset_key}_{measure}_overall.png
  ks_results/plots/{dataset_key}_{lang}_{measure}.png

Usage:
  python "new code\\ks_test_real_vs_random.py"
"""

from __future__ import annotations

import ast
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "ks_results"
SENTENCE_OUT_DIR = OUT_DIR / "sentence_level"
PLOT_OUT_DIR = OUT_DIR / "plots"
HIGH_ARITY_THRESHOLD = 3

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

DATASETS = [
    {
        "key": "matched_random",
        "label": "Matched random structures",
        "input": ROOT / "target_random_structures.csv",
    },
    {
        "key": "pure_random_pruefer",
        "label": "Pure random (Pruefer)",
        "input": ROOT / "pure_random_structures_pruefer.csv",
    },
    {
        "key": "pure_random_root0",
        "label": "Pure random (root 0)",
        "input": ROOT / "pure_random_structures_root0.csv",
    },
]

MEASURES = [
    ("mean_dep_distance", "Mean dependency length"),
    ("mean_intervener_complexity", "Mean intervener complexity"),
    ("tree_depth", "Tree depth"),
    ("tree_density", "Tree density"),
    ("max_arity", "Maximum arity"),
    ("avg_arity", "Average arity"),
    ("proportion_high_arity_nodes", f"Proportion of nodes with arity >= {HIGH_ARITY_THRESHOLD}"),
]

COLORS = {
    "real": "#2f73b7",
    "random": "#c84c4c",
}


def normalize_lang(raw_lang: str) -> str:
    return raw_lang.strip().replace("\\", "/").strip("/").split("/")[-1]


def sentence_nodes(dataset_key: str, kind: str, sent_len: int) -> set[int]:
    if kind == "real":
        return set(range(1, sent_len + 1)) | {0}
    if dataset_key in {"matched_random", "pure_random_pruefer"}:
        return set(range(sent_len)) | {1000}
    return set(range(sent_len + 1))


def tree_density(nodes: set[int], outdegree: Counter) -> float:
    return sum(1 for node in nodes if outdegree[node] > 0) / len(nodes)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def format_p_value(p_value: float) -> str:
    if p_value == 0.0:
        return "<1e-300"
    return f"{p_value:.3e}"


def read_sentence_rows(dataset: dict[str, object]) -> list[dict[str, int | float | str]]:
    infile = Path(dataset["input"])
    if not infile.exists():
        raise SystemExit(f"Missing input dataset: {infile}")

    by_sentence: dict[tuple[str, str, int], dict[str, object]] = {}
    with infile.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if len(parts) != len(COLUMNS):
                continue

            row = dict(zip(COLUMNS, parts))
            lang = normalize_lang(row["lang"])
            kind = row["kind"]
            sent_len = int(row["sent_len"])
            sent_id = int(row["sent_id"])
            key = (lang, kind, sent_id)

            entry = by_sentence.setdefault(
                key,
                {
                    "dataset_key": str(dataset["key"]),
                    "dataset_label": str(dataset["label"]),
                    "lang": lang,
                    "kind": kind,
                    "sent_id": sent_id,
                    "sent_len": sent_len,
                    "max_arity": int(row["max_arity"]),
                    "avg_arity": float(row["avg_arity"]),
                    "tree_depth": int(row["projection_degree"]),
                    "dep_distances": [],
                    "dep_depths": [],
                    "nodes": sentence_nodes(str(dataset["key"]), kind, sent_len),
                    "outdegree": Counter(),
                },
            )

            entry["dep_distances"].append(float(row["dep_distance"]))  # type: ignore[index]
            entry["dep_depths"].append(float(row["dep_depth"]))  # type: ignore[index]

            head, dependent = ast.literal_eval(row["edge"])
            nodes = entry["nodes"]  # type: ignore[assignment]
            outdegree = entry["outdegree"]  # type: ignore[assignment]
            nodes.update([head, dependent])
            outdegree[head] += 1

    sentence_rows: list[dict[str, int | float | str]] = []
    for entry in by_sentence.values():
        dep_distances = entry["dep_distances"]  # type: ignore[assignment]
        dep_depths = entry["dep_depths"]  # type: ignore[assignment]
        nodes = entry["nodes"]  # type: ignore[assignment]
        outdegree = entry["outdegree"]  # type: ignore[assignment]
        high_arity_nodes = sum(1 for node in nodes if outdegree[node] >= HIGH_ARITY_THRESHOLD)

        sentence_rows.append(
            {
                "dataset_key": entry["dataset_key"],
                "dataset_label": entry["dataset_label"],
                "lang": entry["lang"],
                "kind": entry["kind"],
                "sent_id": entry["sent_id"],
                "sent_len": entry["sent_len"],
                "mean_dep_distance": sum(dep_distances) / len(dep_distances),
                "mean_intervener_complexity": sum(dep_depths) / len(dep_depths),
                "tree_depth": entry["tree_depth"],
                "tree_density": tree_density(nodes, outdegree),
                "max_arity": entry["max_arity"],
                "avg_arity": entry["avg_arity"],
                "proportion_high_arity_nodes": high_arity_nodes / len(nodes),
            }
        )

    return sentence_rows


def write_sentence_rows(dataset_key: str, rows: list[dict[str, int | float | str]]) -> None:
    SENTENCE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = SENTENCE_OUT_DIR / f"{dataset_key}_sentence_level_stats.csv"
    with outfile.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_key",
                "dataset_label",
                "lang",
                "kind",
                "sent_id",
                "sent_len",
                "mean_dep_distance",
                "mean_intervener_complexity",
                "tree_depth",
                "tree_density",
                "max_arity",
                "avg_arity",
                "proportion_high_arity_nodes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def ks_row(
    *,
    dataset_key: str,
    dataset_label: str,
    lang: str,
    measure: str,
    real_values: list[float],
    random_values: list[float],
) -> dict[str, int | float | str]:
    result = ks_2samp(real_values, random_values, alternative="two-sided", mode="auto")
    return {
        "dataset_key": dataset_key,
        "dataset_label": dataset_label,
        "lang": lang,
        "measure": measure,
        "n_real": len(real_values),
        "n_random": len(random_values),
        "real_mean": mean(real_values),
        "random_mean": mean(random_values),
        "ks_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "p_value_display": format_p_value(float(result.pvalue)),
    }


def compute_ks_results(
    rows: list[dict[str, int | float | str]],
) -> tuple[list[dict[str, int | float | str]], list[dict[str, int | float | str]]]:
    overall_rows: list[dict[str, int | float | str]] = []
    by_language_rows: list[dict[str, int | float | str]] = []

    dataset_key = str(rows[0]["dataset_key"])
    dataset_label = str(rows[0]["dataset_label"])
    langs = sorted({str(row["lang"]) for row in rows})

    for measure, _ in MEASURES:
        real_values = [float(row[measure]) for row in rows if row["kind"] == "real"]
        random_values = [float(row[measure]) for row in rows if row["kind"] == "random"]
        if real_values and random_values:
            overall_rows.append(
                ks_row(
                    dataset_key=dataset_key,
                    dataset_label=dataset_label,
                    lang="overall",
                    measure=measure,
                    real_values=real_values,
                    random_values=random_values,
                )
            )

        for lang in langs:
            lang_rows = [row for row in rows if row["lang"] == lang]
            real_values = [float(row[measure]) for row in lang_rows if row["kind"] == "real"]
            random_values = [float(row[measure]) for row in lang_rows if row["kind"] == "random"]
            if real_values and random_values:
                by_language_rows.append(
                    ks_row(
                        dataset_key=dataset_key,
                        dataset_label=dataset_label,
                        lang=lang,
                        measure=measure,
                        real_values=real_values,
                        random_values=random_values,
                    )
                )

    return overall_rows, by_language_rows


def ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    xs = sorted(values)
    ys = [(i + 1) / len(xs) for i in range(len(xs))]
    return xs, ys


def plot_distribution(
    *,
    dataset_key: str,
    dataset_label: str,
    measure: str,
    measure_label: str,
    real_values: list[float],
    random_values: list[float],
    scope_label: str,
    outfile: Path,
) -> None:
    fig, (ax_hist, ax_ecdf) = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=180)

    bins = 30
    ax_hist.hist(
        real_values,
        bins=bins,
        alpha=0.55,
        color=COLORS["real"],
        label="Real trees",
        density=True,
    )
    ax_hist.hist(
        random_values,
        bins=bins,
        alpha=0.55,
        color=COLORS["random"],
        label="Random trees",
        density=True,
    )
    ax_hist.set_title(f"{scope_label}: histogram", fontsize=11)
    ax_hist.set_xlabel(measure_label, fontsize=10)
    ax_hist.set_ylabel("Density", fontsize=10)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.legend(frameon=False)

    xs_real, ys_real = ecdf(real_values)
    xs_rand, ys_rand = ecdf(random_values)
    ax_ecdf.step(xs_real, ys_real, where="post", color=COLORS["real"], linewidth=2.0, label="Real trees")
    ax_ecdf.step(
        xs_rand,
        ys_rand,
        where="post",
        color=COLORS["random"],
        linewidth=2.0,
        label="Random trees",
    )
    ax_ecdf.set_title(f"{scope_label}: ECDF", fontsize=11)
    ax_ecdf.set_xlabel(measure_label, fontsize=10)
    ax_ecdf.set_ylabel("Cumulative probability", fontsize=10)
    ax_ecdf.spines["top"].set_visible(False)
    ax_ecdf.spines["right"].set_visible(False)
    ax_ecdf.legend(frameon=False)

    fig.suptitle(f"{dataset_label}: {measure_label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def write_csv(outfile: Path, rows: list[dict[str, int | float | str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_key",
                "dataset_label",
                "lang",
                "measure",
                "n_real",
                "n_random",
                "real_mean",
                "random_mean",
                "ks_statistic",
                "p_value",
                "p_value_display",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SENTENCE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_results: list[dict[str, int | float | str]] = []
    by_language_results: list[dict[str, int | float | str]] = []

    for dataset in DATASETS:
        rows = read_sentence_rows(dataset)
        dataset_key = str(dataset["key"])
        dataset_label = str(dataset["label"])

        write_sentence_rows(dataset_key, rows)
        overall_rows, language_rows = compute_ks_results(rows)
        overall_results.extend(overall_rows)
        by_language_results.extend(language_rows)

        langs = sorted({str(row["lang"]) for row in rows})
        for measure, measure_label in MEASURES:
            real_values = [float(row[measure]) for row in rows if row["kind"] == "real"]
            random_values = [float(row[measure]) for row in rows if row["kind"] == "random"]
            if real_values and random_values:
                plot_distribution(
                    dataset_key=dataset_key,
                    dataset_label=dataset_label,
                    measure=measure,
                    measure_label=measure_label,
                    real_values=real_values,
                    random_values=random_values,
                    scope_label="overall",
                    outfile=PLOT_OUT_DIR / f"{dataset_key}_{measure}_overall.png",
                )

            for lang in langs:
                lang_rows = [row for row in rows if row["lang"] == lang]
                real_values = [float(row[measure]) for row in lang_rows if row["kind"] == "real"]
                random_values = [float(row[measure]) for row in lang_rows if row["kind"] == "random"]
                if real_values and random_values:
                    plot_distribution(
                        dataset_key=dataset_key,
                        dataset_label=dataset_label,
                        measure=measure,
                        measure_label=measure_label,
                        real_values=real_values,
                        random_values=random_values,
                        scope_label=lang,
                        outfile=PLOT_OUT_DIR / f"{dataset_key}_{lang}_{measure}.png",
                    )

    write_csv(OUT_DIR / "overall_ks_results.csv", overall_results)
    write_csv(OUT_DIR / "by_language_ks_results.csv", by_language_results)

    print(f"Wrote {OUT_DIR / 'overall_ks_results.csv'}")
    print(f"Wrote {OUT_DIR / 'by_language_ks_results.csv'}")
    print(f"Wrote sentence-level summaries to {SENTENCE_OUT_DIR}")
    print(f"Wrote plots to {PLOT_OUT_DIR}")


if __name__ == "__main__":
    main()
