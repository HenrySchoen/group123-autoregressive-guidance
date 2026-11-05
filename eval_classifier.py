"""Evaluate family classification accuracy by prefix length.

Run from the repo root:
    uv run python eval_classifier.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from load_data import get_loaders
from protgen.data import FAMILIES
from protgen.models.classifier import FamilyClassifier, valid_positions
from train_classifier import get_device, load_backbone

CKPT_DIR = Path("results/checkpoints")
OUT_DIR = Path("results/classifier")


def load_model(
    backbone_path: Path, head_path: Path, device: torch.device
) -> FamilyClassifier:
    model = FamilyClassifier(load_backbone(backbone_path, device))
    model.head.load_state_dict(
        torch.load(head_path, map_location=device)["head_state_dict"]
    )
    return model.to(device).eval()


@torch.no_grad()
def prefix_accuracy(model, loader, device, max_length: int):
    hits = torch.zeros(len(FAMILIES), max_length)
    counts = torch.zeros(len(FAMILIES), max_length)
    clean_hits = torch.zeros(len(FAMILIES))
    clean_counts = torch.zeros(len(FAMILIES))

    for x, _, y, length in loader:
        x, y, length = x.to(device), y.to(device), length.to(device)
        predicted = model(x).argmax(dim=-1)
        batch, eos = torch.arange(len(x), device=device), length + 1

        mask = valid_positions(x)
        mask[batch, eos] = False  # the EOS state sees the whole sequence, not a prefix
        correct = (predicted == y[:, None]) & mask
        clean_correct = predicted[batch, eos] == y

        for i in range(len(FAMILIES)):
            rows = y == i
            hits[i] += correct[rows].sum(dim=0).cpu()
            # shorter sequences stop early, so every position needs its own count
            counts[i] += mask[rows].sum(dim=0).cpu()
            clean_hits[i] += clean_correct[rows].sum().cpu()
            clean_counts[i] += rows.sum().cpu()

    return hits, counts, clean_hits, clean_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=str(CKPT_DIR / "transformer.pt"))
    parser.add_argument("--checkpoint", default=str(CKPT_DIR / "classifier.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    _, val_data, _, _, val_loader, _ = get_loaders(args.batch_size)
    model = load_model(Path(args.backbone), Path(args.checkpoint), device)
    hits, counts, clean_hits, clean_counts = prefix_accuracy(
        model, val_loader, device, val_data.max_length
    )

    table = pd.DataFrame({"prefix_length": range(val_data.max_length)})
    table["accuracy"] = (hits.sum(0) / counts.sum(0)).numpy()
    table["n_sequences"] = counts.sum(0).numpy()

    for i, family in enumerate(FAMILIES):
        table[family] = (hits[i] / counts[i]).numpy()

    table = table[table["n_sequences"] > 0]

    clean = pd.DataFrame(
        {
            "family": FAMILIES,
            "accuracy": (clean_hits / clean_counts).numpy(),
            "n_sequences": clean_counts.numpy(),
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "prefix_accuracy.csv", index=False)
    clean.to_csv(OUT_DIR / "clean_accuracy.csv", index=False)

    for family in FAMILIES:
        plt.plot(table["prefix_length"], table[family], label=family)

    plt.plot(table["prefix_length"], table["accuracy"], "k--", label="all")
    plt.axhline(1 / len(FAMILIES), color="grey", lw=0.8, label="chance")
    plt.xlabel("prefix length")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(OUT_DIR / "prefix_accuracy.png", dpi=150)

    print("complete sequences")
    print(clean.to_string(index=False))
    print(f"all {(clean_hits.sum() / clean_counts.sum()).item():.4f}\n")
    print("by prefix length")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
