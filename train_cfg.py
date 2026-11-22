"""Train the conditional transformer with label dropout, for classifier-free guidance.

Run from the repo root:
    uv run python train_cfg.py --epochs 20
"""
import argparse
import json
import math

import torch
from tqdm import tqdm

from load_data import get_loaders
from protgen.models.transformer_cfg import ConditionalTransformer, NULL_LABEL
from train import CKPT_DIR, CONFIG, get_device, lm_loss


def drop_labels(y, p_drop):
    # replace the family label with the null label with probability p_drop
    keep = torch.rand(y.shape, device=y.device) >= p_drop
    return torch.where(keep, y, torch.full_like(y, NULL_LABEL))


def train_epoch(model, loader, optimizer, device, p_drop):
    model.train()
    total, n = 0.0, 0
    for x, x_shifted, y, _ in loader:
        x, x_shifted, y = x.to(device), x_shifted.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, drop_labels(y, p_drop))
        loss = lm_loss(logits, x_shifted)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / n


@torch.no_grad()
def evaluate(model, loader, device, conditional=True):
    # conditional=False evaluates the model with the null label
    model.eval()
    total, n = 0.0, 0
    for x, x_shifted, y, _ in loader:
        x, x_shifted, y = x.to(device), x_shifted.to(device), y.to(device)
        logits = model(x, y if conditional else None)
        total += lm_loss(logits, x_shifted).item()
        n += 1
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--p-drop", type=float)
    parser.add_argument("--seed", type=int, default=None)
    parser.set_defaults(**CONFIG["train"], **CONFIG["cfg"])
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device()
    _, _, _, train_loader, val_loader, _ = get_loaders(args.batch_size)

    model = ConditionalTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"cfg transformer on {device}: {sum(p.numel() for p in model.parameters()):,} parameters")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val, history = float("inf"), []
    for epoch in tqdm(range(1, args.epochs + 1), unit="epoch", ncols=80, colour="green"):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.p_drop)
        val_cond = evaluate(model, val_loader, device, conditional=True)
        val_null = evaluate(model, val_loader, device, conditional=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_cond,
                        "val_null_loss": val_null, "val_ppl": math.exp(val_cond)})

        if val_cond < best_val:
            best_val = val_cond
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": val_cond, "p_drop": args.p_drop}, CKPT_DIR / "cfg.pt")

        (CKPT_DIR / "cfg_history.json").write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()