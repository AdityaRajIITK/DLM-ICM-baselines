"""
Construct random-structure baseline output for target languages only.

This is a quiet, resumable version of construct_output_random_structures.py for:
  hi, zh, ja, tr, fi, ar, es

It writes to target_random_structures.csv instead of Random_structures.csv so the
existing English experiment is left untouched.

Usage from the repo root:
  python "new code\\construct_target_random_structures.py"
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import networkx as nx
from conllu import parse

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Measures import Compute_measures
from Measures_rand import Compute_measures_rand
from baseline_conditions_random_structures import Random_base


TARGET_LANGS = ("hi", "zh", "ja", "tr", "fi", "ar", "es")
SUD_DIR = ROOT_DIR / "SUD"
OUTPUT = ROOT_DIR / "target_random_structures.csv"


def existing_completed_sentences() -> set[tuple[str, int]]:
    completed_kinds: dict[tuple[str, int], set[str]] = {}
    if not OUTPUT.exists():
        return set()

    with OUTPUT.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if len(parts) != 17:
                continue
            lang = parts[0].strip().replace("\\", "")
            kind = parts[1]
            sent_id = int(parts[2])
            completed_kinds.setdefault((lang, sent_id), set()).add(kind)

    return {key for key, kinds in completed_kinds.items() if {"real", "random"} <= kinds}


def clean_incomplete_sentences() -> None:
    if not OUTPUT.exists():
        return

    rows = []
    completed_kinds: dict[tuple[str, int], set[str]] = {}
    with OUTPUT.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            rows.append(parts)
            if len(parts) != 17:
                continue
            lang = parts[0].strip().replace("\\", "")
            kind = parts[1]
            sent_id = int(parts[2])
            completed_kinds.setdefault((lang, sent_id), set()).add(kind)

    incomplete = {key for key, kinds in completed_kinds.items() if {"real", "random"} - kinds}
    if not incomplete:
        return

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for parts in rows:
            if len(parts) != 17:
                continue
            lang = parts[0].strip().replace("\\", "")
            sent_id = int(parts[2])
            if (lang, sent_id) not in incomplete:
                writer.writerow(parts)

    print(f"Removed {len(incomplete)} incomplete sentence/tree pairs before resuming.", flush=True)


def token_id_is_int(value) -> bool:
    return isinstance(value, int)


def build_tree(sentence) -> nx.DiGraph:
    tree = nx.DiGraph()
    for nodeinfo in sentence:
        token_id = nodeinfo.get("id")
        head = nodeinfo.get("head")
        deprel = nodeinfo.get("deprel")
        if not token_id_is_int(token_id):
            continue
        if deprel == "punct":
            continue
        tree.add_node(
            token_id,
            form=nodeinfo.get("form"),
            lemma=nodeinfo.get("lemma"),
            upostag=nodeinfo.get("upos"),
            xpostag=nodeinfo.get("xpos"),
            feats=nodeinfo.get("feats"),
            head=head,
            deprel=deprel,
            deps=nodeinfo.get("deps"),
            misc=nodeinfo.get("misc"),
        )

    tree.add_node(0)
    for node in list(tree.nodes):
        if node == 0:
            continue
        head = tree.nodes[node].get("head")
        if tree.has_node(head):
            tree.add_edge(head, node, drel=tree.nodes[node].get("deprel"))

    return tree


def num_content_nodes(tree: nx.DiGraph) -> int:
    return len([node for node in tree.nodes if node != 0])


def is_well_formed_tree(tree: nx.DiGraph) -> bool:
    # For the filtered dependency trees we want one incoming edge per content node.
    return len(tree.edges) == num_content_nodes(tree)


def count_nonprojective_edges(tree: nx.DiGraph, measures: Compute_measures) -> int:
    count = 0
    for edge in tree.edges:
        if edge[0] == 0:
            continue
        if not measures.is_projective(edge):
            count += 1
    return count


def row_values(lang: str, kind: str, sent_id: int, sent_len: int, tree, measures, root: int):
    max_arity = measures.arity()[0]
    avg_arity = measures.arity()[1]
    projection_degree = measures.projection_degree(root)
    gap_degree = measures.gap_degree(root)
    k_illnestedness = measures.illnestedness(root, gap_degree)

    for edge in tree.edges:
        if edge[0] == root:
            continue
        projectivity = 1 if measures.is_projective(edge) else 0
        yield [
            f"\\{lang}",
            kind,
            sent_id,
            sent_len,
            max_arity,
            avg_arity,
            projection_degree,
            gap_degree,
            k_illnestedness,
            edge,
            measures.dependency_direction(edge),
            measures.dependency_distance(edge),
            measures.dependency_depth(edge),
            projectivity,
            measures.edge_degree(edge),
            measures.endpoint_crossing(edge),
            measures.hdd(edge),
        ]


def process_language(lang: str, completed: set[tuple[str, int]]) -> tuple[int, int, int]:
    infile = SUD_DIR / f"{lang}-sud-train.conllu"
    if not infile.exists():
        print(f"{lang}: missing {infile}", flush=True)
        return (0, 0, 0)

    print(f"{lang}: reading {infile}", flush=True)
    sentences = parse(infile.read_text(encoding="utf-8"))
    processed = 0
    skipped = 0
    failed = 0

    with OUTPUT.open("a", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t", lineterminator="\n")

        for sent_id, sentence in enumerate(sentences, start=1):
            if sent_id == 1 or (lang, sent_id) in completed:
                skipped += 1
                continue

            tree = build_tree(sentence)
            sent_len = num_content_nodes(tree)
            if not (1 < sent_len < 12):
                skipped += 1
                continue
            if not is_well_formed_tree(tree):
                skipped += 1
                continue

            try:
                real_measures = Compute_measures(tree)
                num_cross_real = count_nonprojective_edges(tree, real_measures)
                random_trees = Random_base(tree).gen_random(num_cross_real)
                if not random_trees:
                    failed += 1
                    continue

                random_tree = random_trees[0]
                random_measures = Compute_measures_rand(random_tree, 1000)

                for row in row_values(
                    lang, "random", sent_id, sent_len, random_tree, random_measures, 1000
                ):
                    writer.writerow(row)
                for row in row_values(lang, "real", sent_id, sent_len, tree, real_measures, 0):
                    writer.writerow(row)

                processed += 1
            except Exception as exc:
                failed += 1
                print(f"{lang}: failed sentence {sent_id}: {exc}", flush=True)

            if (processed + failed) % 100 == 0:
                print(
                    f"{lang}: processed={processed}, failed={failed}, skipped={skipped}, "
                    f"sent_id={sent_id}",
                    flush=True,
                )

    print(f"{lang}: done processed={processed}, failed={failed}, skipped={skipped}", flush=True)
    return (processed, failed, skipped)


def main() -> None:
    clean_incomplete_sentences()
    completed = existing_completed_sentences()
    if completed:
        print(f"Resuming with {len(completed)} already completed sentence/tree pairs.", flush=True)

    totals = {}
    for lang in TARGET_LANGS:
        totals[lang] = process_language(lang, completed)

    print(f"Wrote {OUTPUT}", flush=True)
    for lang, (processed, failed, skipped) in totals.items():
        print(f"{lang}: processed={processed}, failed={failed}, skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
