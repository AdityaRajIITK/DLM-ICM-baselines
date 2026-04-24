"""
Analyze the Pruefer-based pure-random pipeline outputs.

Inputs:
  pure_random_structures_pruefer.csv

Outputs:
  figures_random/
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_pure_random_languages.py")


def main() -> None:
    os.environ["PURE_RANDOM_INPUT"] = "pure_random_structures_pruefer.csv"
    os.environ["PURE_RANDOM_FIGURES_DIR"] = "figures_random"
    os.environ["PURE_RANDOM_LABEL"] = "Pure random trees (Pruefer)"
    os.environ["PURE_RANDOM_ANALYSIS_MODE"] = "pruefer"
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
