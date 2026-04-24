"""
Construct a pure random-tree baseline with no crossing matching.

For every eligible sentence, this keeps the sentence length fixed and generates
one random directed tree with the same number of nodes, rooted at node 0,
without requiring the random tree to match the real tree's number of crossings.

It writes to `pure_random_structures.csv` and is resumable.

Usage from the repo root:
  python "new code\\construct_pure_random_structures.py"
"""

from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path

import networkx as nx
from conllu import parse

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Measures import Compute_measures
from Measures_rand import Compute_measures_rand
import treegen as gen


SUD_DIR = ROOT_DIR / "SUD"
OUTPUT = ROOT_DIR / os.environ.get("PURE_RANDOM_OUTPUT", "pure_random_structures.csv")
GENERATOR_MODE = os.environ.get("PURE_RANDOM_GENERATOR", "root0").strip().lower()


def available_languages() -> list[str]:
    langs = []
    for path in sorted(SUD_DIR.glob("*-sud-train.conllu")):
        langs.append(path.name.replace("-sud-train.conllu", ""))
    return langs


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
    # After filtering punctuation, keep only sentences whose remaining nodes
    # still form a proper rooted tree.
    return len(tree.edges) == num_content_nodes(tree)


def generate_pure_random_tree_root0(num_nodes: int) -> nx.DiGraph:
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")

    tree = nx.DiGraph()
    tree.add_nodes_from(range(num_nodes))

    for node in range(1, num_nodes):
        parent = random.randrange(0, node)
        tree.add_edge(parent, node)
        tree.nodes[node]["head"] = parent

    tree.nodes[0]["head"] = 0
    return tree


def generate_pure_random_tree_pruefer(sent_len: int) -> nx.DiGraph:
    code = gen.random_pruefer_code(sent_len)
    directed_options = list(gen.directed_trees(gen.tree_from_pruefer_code(code)))
    if not directed_options:
        raise ValueError(f"Could not generate a directed tree for sentence length {sent_len}")
    random.shuffle(directed_options)
    directed = directed_options[0]
    real_root = next(nx.topological_sort(directed))
    abstract_root = 1000
    directed.add_edge(abstract_root, real_root)
    for edge in directed.edges:
        directed.nodes[edge[1]]["head"] = edge[0]
    return directed


def generate_random_tree(sent_len: int) -> tuple[nx.DiGraph, int]:
    if GENERATOR_MODE == "root0":
        return generate_pure_random_tree_root0(sent_len + 1), 0
    if GENERATOR_MODE == "pruefer":
        return generate_pure_random_tree_pruefer(sent_len), 1000
    raise ValueError(f"Unsupported PURE_RANDOM_GENERATOR mode: {GENERATOR_MODE}")


def row_values(lang: str, kind: str, sent_id: int, sent_len: int, tree, measures, root: int):
    max_arity = measures.arity()[0]
    avg_arity = measures.arity()[1]
    projection_degree = measures.projection_degree(root)
    gap_degree = measures.gap_degree(root)
    k_illnestedness = measures.illnestedness(root, gap_degree)

    for edge in tree.edges:
        if edge[0] == root and (kind == "real" or root != 0):
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
                random_tree, random_root = generate_random_tree(sent_len)
                random_measures = Compute_measures_rand(random_tree, random_root)

                for row in row_values(
                    lang, "random", sent_id, sent_len, random_tree, random_measures, random_root
                ):
                    writer.writerow(row)
                for row in row_values(lang, "real", sent_id, sent_len, tree, real_measures, 0):
                    writer.writerow(row)
                processed += 1
            except Exception as exc:
                failed += 1
                print(f"{lang}: failed sentence {sent_id}: {exc}", flush=True)

            if (processed + failed) % 500 == 0:
                print(
                    f"{lang}: processed={processed}, failed={failed}, skipped={skipped}, "
                    f"sent_id={sent_id}",
                    flush=True,
                )

    print(f"{lang}: done processed={processed}, failed={failed}, skipped={skipped}", flush=True)
    return (processed, failed, skipped)


def main() -> None:
    print(
        f"Pure-random generator mode: {GENERATOR_MODE}; output: {OUTPUT.name}",
        flush=True,
    )
    clean_incomplete_sentences()
    completed = existing_completed_sentences()
    if completed:
        print(f"Resuming with {len(completed)} already completed sentence/tree pairs.", flush=True)

    totals = {}
    for lang in available_languages():
        totals[lang] = process_language(lang, completed)

    print(f"Wrote {OUTPUT}", flush=True)
    for lang, (processed, failed, skipped) in totals.items():
        print(f"{lang}: processed={processed}, failed={failed}, skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
