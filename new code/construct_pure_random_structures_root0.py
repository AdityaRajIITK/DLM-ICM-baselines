"""
Run the root-0 pure-random pipeline.

Outputs:
  pure_random_structures_root0.csv
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


SCRIPT = Path(__file__).with_name("construct_pure_random_structures.py")


def main() -> None:
    os.environ["PURE_RANDOM_GENERATOR"] = "root0"
    os.environ["PURE_RANDOM_OUTPUT"] = "pure_random_structures_root0.csv"
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
