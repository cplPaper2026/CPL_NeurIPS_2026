"""CLI entry point for the representative subset selection experiment.

Usage examples (run from this directory)::

    python main.py --method cpl --dataset cifar10 --epochs 20
    python main.py --method bce --dataset cifar10 --epochs 20
    python main.py --method ar  --dataset cifar10 --epochs 20
    python main.py --method bce --eval-only --checkpoint-path runs/bce_cifar10/best.pth
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import run

from config import parse_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
