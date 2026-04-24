"""
Download target Universal Dependencies train splits from Hugging Face.

The old example

    load_dataset("universal_dependencies", "fr_gsd")

does not work with current `datasets` versions because that dataset uses an old
dataset-script format. The maintained Parquet version is:

    load_dataset("commul/universal_dependencies", CONFIG, revision="2.17")

This script exports one train treebank per target language into filenames that
the existing repo scanner recognizes:

    SUD/{lang}-sud-train.conllu

Important: these are UD treebanks from Hugging Face, not the original SUD
converted treebanks used by the paper. The filename is for compatibility with
this repo's existing `construct_output_*.py` scripts.

Usage from the repo root:
    python "new code\\download_ud_from_huggingface.py"
"""

from __future__ import annotations

import csv
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "commul/universal_dependencies"
REVISION = "2.17"
SPLIT = "train"
OUT_DIR = Path("SUD")
REPORT = OUT_DIR / "huggingface_ud_download_report.csv"

TREEBANKS = {
    "hi": ("Hindi", "hi_hdtb"),
    "zh": ("Chinese", "zh_gsd"),
    "ja": ("Japanese", "ja_gsd"),
    "tr": ("Turkish", "tr_imst"),
    "fi": ("Finnish", "fi_tdt"),
    "ar": ("Arabic", "ar_padt"),
    "es": ("Spanish", "es_ancora"),
}


def conllu_value(value) -> str:
    if value is None or value == "":
        return "_"
    return str(value)


def export_dataset_to_conllu(dataset, outfile: Path) -> int:
    sentence_count = 0
    with outfile.open("w", encoding="utf-8", newline="\n") as f:
        for sentence in dataset:
            sentence_count += 1
            sent_id = conllu_value(sentence.get("sent_id"))
            text = conllu_value(sentence.get("text"))
            f.write(f"# sent_id = {sent_id}\n")
            f.write(f"# text = {text}\n")

            tokens = sentence["tokens"]
            lemmas = sentence["lemmas"]
            upos = sentence["upos"]
            xpos = sentence["xpos"]
            feats = sentence["feats"]
            heads = sentence["head"]
            deprels = sentence["deprel"]
            deps = sentence["deps"]
            misc = sentence["misc"]

            for index, token in enumerate(tokens, start=1):
                i = index - 1
                fields = [
                    str(index),
                    conllu_value(token),
                    conllu_value(lemmas[i]),
                    conllu_value(upos[i]),
                    conllu_value(xpos[i]),
                    conllu_value(feats[i]),
                    conllu_value(heads[i]),
                    conllu_value(deprels[i]),
                    conllu_value(deps[i]),
                    conllu_value(misc[i]),
                ]
                f.write("\t".join(fields) + "\n")

            f.write("\n")

    return sentence_count


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    report_rows = []

    for lang, (language_name, config) in TREEBANKS.items():
        outfile = OUT_DIR / f"{lang}-sud-train.conllu"
        print(f"Downloading {language_name} ({lang}) from {config}...")
        dataset = load_dataset(DATASET_ID, config, revision=REVISION, split=SPLIT)
        sentence_count = export_dataset_to_conllu(dataset, outfile)
        report_rows.append(
            {
                "lang": lang,
                "language": language_name,
                "hf_dataset": DATASET_ID,
                "hf_config": config,
                "revision": REVISION,
                "split": SPLIT,
                "sentences": sentence_count,
                "output_file": str(outfile),
            }
        )
        print(f"  wrote {outfile} ({sentence_count} sentences)")

    with REPORT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lang",
                "language",
                "hf_dataset",
                "hf_config",
                "revision",
                "split",
                "sentences",
                "output_file",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
