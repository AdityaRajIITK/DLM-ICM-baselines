"""
Analyze target languages when their SUD train files and random-structure output exist.

Target languages:
  Hindi (hi), Chinese (zh), Japanese (ja), Turkish (tr), Finnish (fi),
  Arabic (ar), and Spanish (es).

This script does not download data or regenerate target_random_structures.csv.
It scans the local SUD directory for files named `{lang}-sud-train.conllu`, then
analyzes the matching rows already present in target_random_structures.csv.

Outputs:
  figures/target_language_availability.csv
  figures/by_language/{lang}_sentence_level_stats.csv
  figures/by_language/{lang}_dependency_length_by_sentence_length.png
  figures/by_language/{lang}_intervener_complexity_by_sentence_length.png
  figures/by_language/{lang}_tree_density_by_sentence_length.png
  figures/by_language/{lang}_max_arity_by_sentence_length.png
  figures/by_language/{lang}_high_arity_nodes_ge3_by_sentence_length.png

Usage from the repo root:
  python "new code\\analyze_target_languages.py"
"""

from __future__ import annotations

import ast
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_LANGS = {
    "hi": "Hindi",
    "zh": "Chinese",
    "ja": "Japanese",
    "tr": "Turkish",
    "fi": "Finnish",
    "ar": "Arabic",
    "es": "Spanish",
}

INPUT = Path("target_random_structures.csv")
SUD_DIR = Path("SUD")
OUT_DIR = Path("figures")
LANG_OUT_DIR = OUT_DIR / "by_language"
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

COLORS = {
    "real": "#2f73b7",
    "random": "#c84c4c",
}

LABELS = {
    "real": "Real trees",
    "random": "Random structures",
}


def normalize_lang(raw_lang: str) -> str:
    return raw_lang.strip().replace("\\", "/").strip("/").split("/")[-1]


def target_sud_files() -> dict[str, Path | None]:
    files: dict[str, Path | None] = {}
    for lang in TARGET_LANGS:
        matches = sorted(SUD_DIR.rglob(f"{lang}-sud-train.conllu"))
        files[lang] = matches[0] if matches else None
    return files


def sentence_nodes(kind: str, sent_len: int) -> set[int]:
    if kind == "random":
        return set(range(sent_len)) | {1000}
    return set(range(1, sent_len + 1)) | {0}


def tree_density(nodes: set[int], outdegree: Counter) -> float:
    # For trees, ordinary graph density is fixed by sentence length, so use the
    # share of nodes that act as branching points.
    return sum(1 for node in nodes if outdegree[node] > 0) / len(nodes)


def read_target_sentence_rows() -> list[dict[str, int | float | str]]:
    if not INPUT.exists():
        raise SystemExit(
            f"{INPUT} not found. Run `python \"new code\\construct_target_random_structures.py\"` first."
        )

    by_sentence: dict[tuple[str, str, int], dict[str, object]] = {}

    with INPUT.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if len(parts) != len(COLUMNS):
                continue

            row = dict(zip(COLUMNS, parts))
            lang = normalize_lang(row["lang"])
            if lang not in TARGET_LANGS:
                continue

            kind = row["kind"]
            sent_len = int(row["sent_len"])
            sent_id = int(row["sent_id"])
            key = (lang, kind, sent_id)
            entry = by_sentence.setdefault(
                key,
                {
                    "lang": lang,
                    "kind": kind,
                    "sent_id": sent_id,
                    "sent_len": sent_len,
                    "max_arity": int(row["max_arity"]),
                    "avg_arity": float(row["avg_arity"]),
                    "dep_distances": [],
                    "dep_depths": [],
                    "nodes": sentence_nodes(kind, sent_len),
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
                "lang": entry["lang"],
                "kind": entry["kind"],
                "sent_id": entry["sent_id"],
                "sent_len": entry["sent_len"],
                "mean_dep_distance": sum(dep_distances) / len(dep_distances),
                "mean_intervener_complexity": sum(dep_depths) / len(dep_depths),
                "tree_density": tree_density(nodes, outdegree),
                "max_arity": entry["max_arity"],
                "avg_arity": entry["avg_arity"],
                "proportion_high_arity_nodes": high_arity_nodes / len(nodes),
            }
        )

    return sentence_rows


def summarize_by_length(
    rows: list[dict[str, int | float | str]]
) -> list[dict[str, int | float | str]]:
    grouped: dict[tuple[str, int, str], list[dict[str, int | float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["lang"]), int(row["sent_len"]), str(row["kind"]))].append(row)

    summary_rows: list[dict[str, int | float | str]] = []
    for (lang, sent_len, kind), group in sorted(grouped.items()):
        summary_rows.append(
            {
                "lang": lang,
                "sent_len": sent_len,
                "kind": kind,
                "num_trees": len(group),
                "mean_dep_distance": mean(float(row["mean_dep_distance"]) for row in group),
                "mean_intervener_complexity": mean(
                    float(row["mean_intervener_complexity"]) for row in group
                ),
                "mean_tree_density": mean(float(row["tree_density"]) for row in group),
                "mean_max_arity": mean(float(row["max_arity"]) for row in group),
                "mean_high_arity_node_proportion": mean(
                    float(row["proportion_high_arity_nodes"]) for row in group
                ),
            }
        )

    return summary_rows


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def write_availability(
    sud_files: dict[str, Path | None],
    sentence_rows: list[dict[str, int | float | str]],
) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    row_counts = Counter(str(row["lang"]) for row in sentence_rows)
    with (OUT_DIR / "target_language_availability.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lang",
                "language",
                "sud_train_file_found",
                "sud_train_file",
                "sentence_level_rows_in_random_structures",
            ],
        )
        writer.writeheader()
        for lang, name in TARGET_LANGS.items():
            sud_file = sud_files[lang]
            writer.writerow(
                {
                    "lang": lang,
                    "language": name,
                    "sud_train_file_found": bool(sud_file),
                    "sud_train_file": str(sud_file) if sud_file else "",
                    "sentence_level_rows_in_random_structures": row_counts[lang],
                }
            )


def write_language_stats(lang: str, rows: list[dict[str, int | float | str]]) -> None:
    LANG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = LANG_OUT_DIR / f"{lang}_sentence_level_stats.csv"
    with outfile.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lang",
                "sent_len",
                "kind",
                "num_trees",
                "mean_dep_distance",
                "mean_intervener_complexity",
                "mean_tree_density",
                "mean_max_arity",
                "mean_high_arity_node_proportion",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_measure(
    lang: str,
    rows: list[dict[str, int | float | str]],
    *,
    measure: str,
    ylabel: str,
    title: str,
    outfile: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=180)

    for kind in ("real", "random"):
        kind_rows = [row for row in rows if row["kind"] == kind]
        kind_rows.sort(key=lambda row: int(row["sent_len"]))
        if not kind_rows:
            continue
        ax.plot(
            [int(row["sent_len"]) for row in kind_rows],
            [float(row[measure]) for row in kind_rows],
            color=COLORS[kind],
            marker="o",
            markersize=4,
            linewidth=2.2,
            label=LABELS[kind],
        )

    ax.set_title(f"{TARGET_LANGS[lang]} ({lang}): {title}", fontsize=12, pad=12)
    ax.set_xlabel("Sentence length", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def plot_language(lang: str, rows: list[dict[str, int | float | str]]) -> None:
    LANG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_measure(
        lang,
        rows,
        measure="mean_dep_distance",
        ylabel="Mean dependency length",
        title="dependency length by sentence length",
        outfile=LANG_OUT_DIR / f"{lang}_dependency_length_by_sentence_length.png",
    )
    plot_measure(
        lang,
        rows,
        measure="mean_intervener_complexity",
        ylabel="Mean intervener complexity",
        title="intervener complexity by sentence length",
        outfile=LANG_OUT_DIR / f"{lang}_intervener_complexity_by_sentence_length.png",
    )
    plot_measure(
        lang,
        rows,
        measure="mean_tree_density",
        ylabel="Mean tree density",
        title="tree density by sentence length",
        outfile=LANG_OUT_DIR / f"{lang}_tree_density_by_sentence_length.png",
    )
    plot_measure(
        lang,
        rows,
        measure="mean_max_arity",
        ylabel="Mean maximum arity",
        title="maximum arity by sentence length",
        outfile=LANG_OUT_DIR / f"{lang}_max_arity_by_sentence_length.png",
    )
    plot_measure(
        lang,
        rows,
        measure="mean_high_arity_node_proportion",
        ylabel=f"Proportion of nodes with arity >= {HIGH_ARITY_THRESHOLD}",
        title=f"high-arity nodes by sentence length",
        outfile=LANG_OUT_DIR / f"{lang}_high_arity_nodes_ge3_by_sentence_length.png",
    )


def main() -> None:
    sud_files = target_sud_files()
    sentence_rows = read_target_sentence_rows()
    summary_rows = summarize_by_length(sentence_rows)
    write_availability(sud_files, sentence_rows)

    plotted = []
    for lang in TARGET_LANGS:
        rows = [row for row in summary_rows if row["lang"] == lang]
        if not rows:
            continue
        write_language_stats(lang, rows)
        plot_language(lang, rows)
        plotted.append(lang)

    print(f"Wrote {OUT_DIR / 'target_language_availability.csv'}")
    if plotted:
        print("Generated per-language analyses for: " + ", ".join(plotted))
    else:
        print(f"No target-language rows were found in {INPUT}.")
        print("Run the target constructor, then rerun this script.")


if __name__ == "__main__":
    main()
