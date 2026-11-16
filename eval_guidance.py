"""Compare amino acid composition of guided samples across lambda values.

Run from the repo root after guidance_classifier.py:
    uv run python eval_guidance.py --family AcyP
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from protgen.data import FAMILIES
from protgen.metrics.composition import family_means, family_metrics

SEQ_DIR = Path("results/sequences")
OUT_DIR = Path("results/guidance")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="AcyP", choices=FAMILIES)
    parser.add_argument("--reference", default="data/train.csv")
    args = parser.parse_args()

    paths = sorted(SEQ_DIR.glob(f"guided_{args.family}_lmbda*.csv"))
    if not paths:
        raise SystemExit(f"no guided samples for {args.family} in {SEQ_DIR}")

    means = family_means(args.reference)
    tables = []

    for path in paths:
        samples = pd.read_csv(path)
        table = family_metrics(samples["sequence"], means)
        table["lmbda"] = samples["lmbda"]
        tables.append(table)

    summary = pd.concat(tables, ignore_index=True).groupby("lmbda").mean()
    print(summary.round(3).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / f"composition_{args.family}.csv")

    summary[[f"l1_{family}" for family in FAMILIES]].plot(marker="o")
    plt.xlabel("lambda")
    plt.ylabel("L1 distance to family mean (percentage points)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"composition_{args.family}.png", dpi=150)


if __name__ == "__main__":
    main()
