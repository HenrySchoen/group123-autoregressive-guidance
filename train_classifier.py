"""Train the family classifier on a frozen transformer.

Run from the repo root:
    uv run python train_classifier.py --epochs 10
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from load_data import get_loaders
from protgen.data import IGNORE_INDEX
from protgen.models.classifier import FamilyClassifier, family_targets
from protgen.models.transformer import Transformer

CKPT_DIR = Path("results/checkpoints")


# duplicated from train.py, which pulls in wandb at import time
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_backbone(path: Path, device: torch.device) -> Transformer:
    checkpoint = torch.load(path, map_location=device)
    backbone = Transformer()
    backbone.load_state_dict(checkpoint["model_state_dict"])
    return backbone


def family_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Cross-entropy expects the class dimension at index 1
    return F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=IGNORE_INDEX)


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, _, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = family_loss(model(x), family_targets(x, y))
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / n


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, _, y, _ in loader:
        x, y = x.to(device), y.to(device)
        total += family_loss(model(x), family_targets(x, y)).item()
        n += 1
    return total / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=str(CKPT_DIR / "transformer.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = get_device()
    _, _, _, train_loader, val_loader, _ = get_loaders(args.batch_size)

    backbone = load_backbone(Path(args.backbone), device)
    model = FamilyClassifier(backbone).to(device)

    # Backbone is frozen, so only the head needs an optimizer
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.head.parameters())
    print(f"classifier on {device}: {n_params:,} trainable parameters")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val, history = float("inf"), []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            checkpoint = {
                "epoch": epoch,
                "head_state_dict": model.head.state_dict(),
                "val_loss": val_loss,
            }
            torch.save(checkpoint, CKPT_DIR / "classifier.pt")

        (CKPT_DIR / "classifier_history.json").write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
