"""
Organize generated statistical-result plots by algorithm and then by language.

This script reorganizes:
  ks_results/plots
  inference_results/plots/distribution
  inference_results/plots/regression

into a nested layout:
  <root>/<algorithm>/<language>/<plot files>

where `overall` is treated like a language bucket.

Usage:
  python "new code\\organize_stat_result_plots.py"
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ALGORITHMS = [
    "matched_random",
    "pure_random_pruefer",
    "pure_random_root0",
]


def split_filename(stem: str) -> tuple[str, str, str] | None:
    for algorithm in ALGORITHMS:
        prefix = f"{algorithm}_"
        if not stem.startswith(prefix):
            continue
        remainder = stem[len(prefix) :]
        if remainder.endswith("_overall"):
            measure = remainder[: -len("_overall")]
            return algorithm, "overall", measure
        parts = remainder.split("_", 1)
        if len(parts) != 2:
            return None
        language, measure = parts
        return algorithm, language, measure
    return None


def organize_flat_plot_dir(plot_dir: Path) -> int:
    if not plot_dir.exists():
        return 0

    moved = 0
    files = [path for path in plot_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"]
    for path in files:
        parsed = split_filename(path.stem)
        if parsed is None:
            continue
        algorithm, language, _ = parsed
        target_dir = plot_dir / algorithm / language
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / path.name
        shutil.move(str(path), str(target_path))
        moved += 1

    return moved


def main() -> None:
    ks_plot_dir = ROOT / "ks_results" / "plots"
    inf_dist_dir = ROOT / "inference_results" / "plots" / "distribution"
    inf_reg_dir = ROOT / "inference_results" / "plots" / "regression"

    moved_ks = organize_flat_plot_dir(ks_plot_dir)
    moved_dist = organize_flat_plot_dir(inf_dist_dir)
    moved_reg = organize_flat_plot_dir(inf_reg_dir)

    print(f"Moved {moved_ks} KS plots into nested folders under {ks_plot_dir}")
    print(f"Moved {moved_dist} inference distribution plots into nested folders under {inf_dist_dir}")
    print(f"Moved {moved_reg} inference regression plots into nested folders under {inf_reg_dir}")


if __name__ == "__main__":
    main()
