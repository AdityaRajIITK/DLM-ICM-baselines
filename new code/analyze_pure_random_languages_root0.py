"""
Analyze the root-0 pure-random pipeline outputs.

Inputs:
  pure_random_structures_root0.csv

Outputs:
  figures_pure_random/
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_pure_random_languages.py")


def main() -> None:
    os.environ["PURE_RANDOM_INPUT"] = "pure_random_structures_root0.csv"
    os.environ["PURE_RANDOM_FIGURES_DIR"] = "figures_pure_random"
    os.environ["PURE_RANDOM_LABEL"] = "Pure random trees (root 0)"
    os.environ["PURE_RANDOM_ANALYSIS_MODE"] = "root0"
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
