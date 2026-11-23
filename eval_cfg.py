"""Compare amino acid composition of classifier-free samples across guidance strengths.

Run from the repo root:
    uv run python eval_cfg.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from protgen.data import FAMILIES
from protgen.metrics.composition import family_means, family_metrics

SEQ_PATH = Path("results/sequences/guided_cfg.csv")
OUT_DIR = Path("results/guidance")


def strength(setting: str) -> float:
    """Map the null label run to -1 so it sorts before w=0."""
    return -1.0 if setting == "null_label" else float(setting.removeprefix("w="))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=str(SEQ_PATH))
    parser.add_argument("--reference", default="data/train.csv")
    args = parser.parse_args()

    samples = pd.read_csv(args.samples)
    sequences = samples["sequence"].fillna("")

    # An empty sequence has no composition at all, and summing its NaN row gives a distance of 0
    kept = sequences.str.len() > 0
    if not kept.all():
        print(f"dropped {(~kept).sum()} empty sequences")

    means = family_means(args.reference)
    table = family_metrics(sequences[kept], means)
    table["w"] = samples["setting"][kept].map(strength)

    summary = table.groupby("w").mean()
    print(summary.round(3).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "composition_cfg.csv")

    summary[[f"l1_{family}" for family in FAMILIES]].plot(marker="o")
    plt.xlabel("guidance strength (-1 is the null label run)")
    plt.ylabel("L1 distance to family mean (percentage points)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "composition_cfg.png", dpi=150)


if __name__ == "__main__":
    main()
