"""
Run Mann-Whitney U, permutation, and regression-comparison tests for real vs random trees.

This script creates new output files only. It reads the available edge-level
datasets, reconstructs sentence-level rows, compares real and random trees with:

- Mann-Whitney U tests on sentence-level distributions
- Permutation tests on sentence-level mean differences
- Regression comparisons for tree depth vs sentence length

Outputs:
  inference_results/distribution_tests_overall.csv
  inference_results/distribution_tests_by_language.csv
  inference_results/regression_tests_overall.csv
  inference_results/regression_tests_by_language.csv
  inference_results/sentence_level/{dataset_key}_sentence_level_stats.csv
  inference_results/plots/distribution/*.png
  inference_results/plots/regression/*.png

Usage:
  python "new code\\mwu_permutation_regression_real_vs_random.py"
"""

from __future__ import annotations

import ast
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "inference_results"
SENTENCE_OUT_DIR = OUT_DIR / "sentence_level"
DISTRIBUTION_PLOT_DIR = OUT_DIR / "plots" / "distribution"
REGRESSION_PLOT_DIR = OUT_DIR / "plots" / "regression"

HIGH_ARITY_THRESHOLD = 3
RNG_SEED = 20260422
PERMUTATION_ITERATIONS = 5000
MAX_PERMUTATION_SAMPLE_PER_GROUP = 4000

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


def mean(values) -> float:
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


def maybe_sample(values: list[float], rng: np.random.Generator) -> tuple[np.ndarray, int]:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= MAX_PERMUTATION_SAMPLE_PER_GROUP:
        return arr, len(arr)
    idx = rng.choice(len(arr), size=MAX_PERMUTATION_SAMPLE_PER_GROUP, replace=False)
    return arr[idx], MAX_PERMUTATION_SAMPLE_PER_GROUP


def permutation_test_mean_difference(
    real_values: list[float],
    random_values: list[float],
    rng: np.random.Generator,
) -> tuple[float, float, int, int]:
    real_sample, n_real_used = maybe_sample(real_values, rng)
    random_sample, n_random_used = maybe_sample(random_values, rng)

    observed = float(real_sample.mean() - random_sample.mean())
    combined = np.concatenate([real_sample, random_sample])
    n_real = len(real_sample)
    count = 0

    for _ in range(PERMUTATION_ITERATIONS):
        shuffled = rng.permutation(combined)
        diff = float(shuffled[:n_real].mean() - shuffled[n_real:].mean())
        if abs(diff) >= abs(observed):
            count += 1

    p_value = (count + 1) / (PERMUTATION_ITERATIONS + 1)
    return observed, p_value, n_real_used, n_random_used


def distribution_test_row(
    *,
    dataset_key: str,
    dataset_label: str,
    lang: str,
    measure: str,
    real_values: list[float],
    random_values: list[float],
    rng: np.random.Generator,
) -> dict[str, int | float | str]:
    mwu = mannwhitneyu(real_values, random_values, alternative="two-sided")
    perm_diff, perm_p, n_real_used, n_random_used = permutation_test_mean_difference(
        real_values, random_values, rng
    )
    return {
        "dataset_key": dataset_key,
        "dataset_label": dataset_label,
        "lang": lang,
        "measure": measure,
        "n_real": len(real_values),
        "n_random": len(random_values),
        "real_mean": mean(real_values),
        "random_mean": mean(random_values),
        "mwu_statistic": float(mwu.statistic),
        "mwu_p_value": float(mwu.pvalue),
        "mwu_p_value_display": format_p_value(float(mwu.pvalue)),
        "perm_mean_difference": float(perm_diff),
        "perm_p_value": float(perm_p),
        "perm_p_value_display": format_p_value(float(perm_p)),
        "perm_n_real_used": n_real_used,
        "perm_n_random_used": n_random_used,
    }


def summarize_depth_by_length(
    rows: list[dict[str, int | float | str]],
) -> dict[tuple[str, str], list[dict[str, int | float | str]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, int | float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["lang"]), str(row["kind"]), int(row["sent_len"]))].append(row)

    summary: dict[tuple[str, str], list[dict[str, int | float | str]]] = defaultdict(list)
    for (lang, kind, sent_len), group in sorted(grouped.items()):
        summary[(lang, kind)].append(
            {
                "lang": lang,
                "kind": kind,
                "sent_len": sent_len,
                "mean_tree_depth": mean(float(row["tree_depth"]) for row in group),
                "num_trees": len(group),
            }
        )
    return summary


def slope_difference_permutation(
    xs_real: np.ndarray,
    ys_real: np.ndarray,
    xs_random: np.ndarray,
    ys_random: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    observed_real = linregress(xs_real, ys_real)
    observed_random = linregress(xs_random, ys_random)
    observed_diff = float(observed_real.slope - observed_random.slope)

    xs = np.concatenate([xs_real, xs_random])
    ys = np.concatenate([ys_real, ys_random])
    n_real = len(xs_real)
    count = 0

    for _ in range(PERMUTATION_ITERATIONS):
        perm = rng.permutation(len(xs))
        real_idx = perm[:n_real]
        rand_idx = perm[n_real:]
        perm_real = linregress(xs[real_idx], ys[real_idx])
        perm_rand = linregress(xs[rand_idx], ys[rand_idx])
        diff = float(perm_real.slope - perm_rand.slope)
        if abs(diff) >= abs(observed_diff):
            count += 1

    p_value = (count + 1) / (PERMUTATION_ITERATIONS + 1)
    return observed_diff, p_value


def regression_test_row(
    *,
    dataset_key: str,
    dataset_label: str,
    lang: str,
    real_rows: list[dict[str, int | float | str]],
    random_rows: list[dict[str, int | float | str]],
    rng: np.random.Generator,
) -> dict[str, int | float | str]:
    xs_real = np.asarray([int(row["sent_len"]) for row in real_rows], dtype=float)
    ys_real = np.asarray([float(row["mean_tree_depth"]) for row in real_rows], dtype=float)
    xs_random = np.asarray([int(row["sent_len"]) for row in random_rows], dtype=float)
    ys_random = np.asarray([float(row["mean_tree_depth"]) for row in random_rows], dtype=float)

    reg_real = linregress(xs_real, ys_real)
    reg_random = linregress(xs_random, ys_random)
    slope_diff, slope_perm_p = slope_difference_permutation(
        xs_real, ys_real, xs_random, ys_random, rng
    )

    return {
        "dataset_key": dataset_key,
        "dataset_label": dataset_label,
        "lang": lang,
        "n_real_points": len(real_rows),
        "n_random_points": len(random_rows),
        "real_slope": float(reg_real.slope),
        "random_slope": float(reg_random.slope),
        "real_intercept": float(reg_real.intercept),
        "random_intercept": float(reg_random.intercept),
        "real_rvalue": float(reg_real.rvalue),
        "random_rvalue": float(reg_random.rvalue),
        "real_regression_p_value": float(reg_real.pvalue),
        "real_regression_p_value_display": format_p_value(float(reg_real.pvalue)),
        "random_regression_p_value": float(reg_random.pvalue),
        "random_regression_p_value_display": format_p_value(float(reg_random.pvalue)),
        "slope_difference": float(slope_diff),
        "intercept_difference": float(reg_real.intercept - reg_random.intercept),
        "slope_permutation_p_value": float(slope_perm_p),
        "slope_permutation_p_value_display": format_p_value(float(slope_perm_p)),
    }


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
    fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=180)

    box = ax_box.boxplot(
        [real_values, random_values],
        tick_labels=["Real", "Random"],
        patch_artist=True,
        widths=0.55,
    )
    box["boxes"][0].set(facecolor=COLORS["real"], alpha=0.65)
    box["boxes"][1].set(facecolor=COLORS["random"], alpha=0.65)
    ax_box.set_title(f"{scope_label}: boxplot", fontsize=11)
    ax_box.set_ylabel(measure_label, fontsize=10)
    ax_box.spines["top"].set_visible(False)
    ax_box.spines["right"].set_visible(False)

    ax_hist.hist(real_values, bins=30, density=True, alpha=0.55, color=COLORS["real"], label="Real")
    ax_hist.hist(
        random_values,
        bins=30,
        density=True,
        alpha=0.55,
        color=COLORS["random"],
        label="Random",
    )
    ax_hist.set_title(f"{scope_label}: histogram", fontsize=11)
    ax_hist.set_xlabel(measure_label, fontsize=10)
    ax_hist.set_ylabel("Density", fontsize=10)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.legend(frameon=False)

    fig.suptitle(f"{dataset_label}: {measure_label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def plot_regression(
    *,
    dataset_label: str,
    scope_label: str,
    real_rows: list[dict[str, int | float | str]],
    random_rows: list[dict[str, int | float | str]],
    outfile: Path,
) -> None:
    xs_real = np.asarray([int(row["sent_len"]) for row in real_rows], dtype=float)
    ys_real = np.asarray([float(row["mean_tree_depth"]) for row in real_rows], dtype=float)
    xs_random = np.asarray([int(row["sent_len"]) for row in random_rows], dtype=float)
    ys_random = np.asarray([float(row["mean_tree_depth"]) for row in random_rows], dtype=float)

    reg_real = linregress(xs_real, ys_real)
    reg_random = linregress(xs_random, ys_random)

    x_min = int(min(xs_real.min(), xs_random.min()))
    x_max = int(max(xs_real.max(), xs_random.max()))
    line_x = np.arange(x_min, x_max + 1, dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=180)
    ax.scatter(xs_real, ys_real, color=COLORS["real"], s=28, alpha=0.9, label="Real means")
    ax.scatter(xs_random, ys_random, color=COLORS["random"], s=28, alpha=0.9, label="Random means")
    ax.plot(line_x, reg_real.intercept + reg_real.slope * line_x, color=COLORS["real"], linewidth=2.2)
    ax.plot(
        line_x,
        reg_random.intercept + reg_random.slope * line_x,
        color=COLORS["random"],
        linewidth=2.2,
    )
    ax.set_title(f"{dataset_label}: {scope_label}", fontsize=12, pad=10)
    ax.set_xlabel("Sentence length", fontsize=11)
    ax.set_ylabel("Mean tree depth", fontsize=11)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def write_csv(outfile: Path, fieldnames: list[str], rows: list[dict[str, int | float | str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SENTENCE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DISTRIBUTION_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    REGRESSION_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    distribution_overall_rows: list[dict[str, int | float | str]] = []
    distribution_language_rows: list[dict[str, int | float | str]] = []
    regression_overall_rows: list[dict[str, int | float | str]] = []
    regression_language_rows: list[dict[str, int | float | str]] = []

    for dataset_index, dataset in enumerate(DATASETS):
        rng = np.random.default_rng(RNG_SEED + dataset_index)
        rows = read_sentence_rows(dataset)
        dataset_key = str(dataset["key"])
        dataset_label = str(dataset["label"])
        langs = sorted({str(row["lang"]) for row in rows})

        write_sentence_rows(dataset_key, rows)

        for measure, measure_label in MEASURES:
            real_values = [float(row[measure]) for row in rows if row["kind"] == "real"]
            random_values = [float(row[measure]) for row in rows if row["kind"] == "random"]
            if real_values and random_values:
                distribution_overall_rows.append(
                    distribution_test_row(
                        dataset_key=dataset_key,
                        dataset_label=dataset_label,
                        lang="overall",
                        measure=measure,
                        real_values=real_values,
                        random_values=random_values,
                        rng=rng,
                    )
                )
                plot_distribution(
                    dataset_key=dataset_key,
                    dataset_label=dataset_label,
                    measure=measure,
                    measure_label=measure_label,
                    real_values=real_values,
                    random_values=random_values,
                    scope_label="overall",
                    outfile=DISTRIBUTION_PLOT_DIR / f"{dataset_key}_{measure}_overall.png",
                )

            for lang in langs:
                lang_rows = [row for row in rows if row["lang"] == lang]
                real_values = [float(row[measure]) for row in lang_rows if row["kind"] == "real"]
                random_values = [float(row[measure]) for row in lang_rows if row["kind"] == "random"]
                if real_values and random_values:
                    distribution_language_rows.append(
                        distribution_test_row(
                            dataset_key=dataset_key,
                            dataset_label=dataset_label,
                            lang=lang,
                            measure=measure,
                            real_values=real_values,
                            random_values=random_values,
                            rng=rng,
                        )
                    )
                    plot_distribution(
                        dataset_key=dataset_key,
                        dataset_label=dataset_label,
                        measure=measure,
                        measure_label=measure_label,
                        real_values=real_values,
                        random_values=random_values,
                        scope_label=lang,
                        outfile=DISTRIBUTION_PLOT_DIR / f"{dataset_key}_{lang}_{measure}.png",
                    )

        depth_summary = summarize_depth_by_length(rows)
        overall_real = []
        overall_random = []
        for (lang, kind), summary_rows in depth_summary.items():
            if kind == "real":
                overall_real.extend(summary_rows)
            elif kind == "random":
                overall_random.extend(summary_rows)

        # Re-average overall by sentence length so each length contributes once.
        def collapse_overall(summary_rows: list[dict[str, int | float | str]], kind: str):
            grouped = defaultdict(list)
            for row in summary_rows:
                grouped[int(row["sent_len"])].append(float(row["mean_tree_depth"]))
            return [
                {"kind": kind, "sent_len": sent_len, "mean_tree_depth": mean(values)}
                for sent_len, values in sorted(grouped.items())
            ]

        overall_real_rows = collapse_overall(overall_real, "real")
        overall_random_rows = collapse_overall(overall_random, "random")
        regression_overall_rows.append(
            regression_test_row(
                dataset_key=dataset_key,
                dataset_label=dataset_label,
                lang="overall",
                real_rows=overall_real_rows,
                random_rows=overall_random_rows,
                rng=rng,
            )
        )
        plot_regression(
            dataset_label=dataset_label,
            scope_label="overall depth vs length",
            real_rows=overall_real_rows,
            random_rows=overall_random_rows,
            outfile=REGRESSION_PLOT_DIR / f"{dataset_key}_overall_tree_depth_vs_length.png",
        )

        for lang in langs:
            real_rows = depth_summary.get((lang, "real"), [])
            random_rows = depth_summary.get((lang, "random"), [])
            if real_rows and random_rows:
                regression_language_rows.append(
                    regression_test_row(
                        dataset_key=dataset_key,
                        dataset_label=dataset_label,
                        lang=lang,
                        real_rows=real_rows,
                        random_rows=random_rows,
                        rng=rng,
                    )
                )
                plot_regression(
                    dataset_label=dataset_label,
                    scope_label=f"{lang} depth vs length",
                    real_rows=real_rows,
                    random_rows=random_rows,
                    outfile=REGRESSION_PLOT_DIR / f"{dataset_key}_{lang}_tree_depth_vs_length.png",
                )

    write_csv(
        OUT_DIR / "distribution_tests_overall.csv",
        [
            "dataset_key",
            "dataset_label",
            "lang",
            "measure",
            "n_real",
            "n_random",
            "real_mean",
            "random_mean",
            "mwu_statistic",
            "mwu_p_value",
            "mwu_p_value_display",
            "perm_mean_difference",
            "perm_p_value",
            "perm_p_value_display",
            "perm_n_real_used",
            "perm_n_random_used",
        ],
        distribution_overall_rows,
    )
    write_csv(
        OUT_DIR / "distribution_tests_by_language.csv",
        [
            "dataset_key",
            "dataset_label",
            "lang",
            "measure",
            "n_real",
            "n_random",
            "real_mean",
            "random_mean",
            "mwu_statistic",
            "mwu_p_value",
            "mwu_p_value_display",
            "perm_mean_difference",
            "perm_p_value",
            "perm_p_value_display",
            "perm_n_real_used",
            "perm_n_random_used",
        ],
        distribution_language_rows,
    )
    write_csv(
        OUT_DIR / "regression_tests_overall.csv",
        [
            "dataset_key",
            "dataset_label",
            "lang",
            "n_real_points",
            "n_random_points",
            "real_slope",
            "random_slope",
            "real_intercept",
            "random_intercept",
            "real_rvalue",
            "random_rvalue",
            "real_regression_p_value",
            "real_regression_p_value_display",
            "random_regression_p_value",
            "random_regression_p_value_display",
            "slope_difference",
            "intercept_difference",
            "slope_permutation_p_value",
            "slope_permutation_p_value_display",
        ],
        regression_overall_rows,
    )
    write_csv(
        OUT_DIR / "regression_tests_by_language.csv",
        [
            "dataset_key",
            "dataset_label",
            "lang",
            "n_real_points",
            "n_random_points",
            "real_slope",
            "random_slope",
            "real_intercept",
            "random_intercept",
            "real_rvalue",
            "random_rvalue",
            "real_regression_p_value",
            "real_regression_p_value_display",
            "random_regression_p_value",
            "random_regression_p_value_display",
            "slope_difference",
            "intercept_difference",
            "slope_permutation_p_value",
            "slope_permutation_p_value_display",
        ],
        regression_language_rows,
    )

    print(f"Wrote {OUT_DIR / 'distribution_tests_overall.csv'}")
    print(f"Wrote {OUT_DIR / 'distribution_tests_by_language.csv'}")
    print(f"Wrote {OUT_DIR / 'regression_tests_overall.csv'}")
    print(f"Wrote {OUT_DIR / 'regression_tests_by_language.csv'}")
    print(f"Wrote sentence-level summaries to {SENTENCE_OUT_DIR}")
    print(f"Wrote distribution plots to {DISTRIBUTION_PLOT_DIR}")
    print(f"Wrote regression plots to {REGRESSION_PLOT_DIR}")


if __name__ == "__main__":
    main()
