"""Train an autoregressive model on the protein splits.

Run from the repo root:
    uv run python train.py --model transformer --epochs 20
    uv run python train.py --model transformer --overfit
"""
import argparse
import json
import math
from pathlib import Path
from tqdm import tqdm
import wandb


import torch
import torch.nn.functional as F

from load_data import get_loaders
from protgen.config import CONFIG
from protgen.models.transformer import Transformer
from protgen.models.mlp import MLP

CKPT_DIR = Path("results/checkpoints")

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(name: str, dropout: float | None = None) -> torch.nn.Module:
    extra = {} if dropout is None else {"dropout": dropout}
    if name == "transformer":
        return Transformer(**{**CONFIG["transformer"], **extra})
    if name == "mlp":
        return MLP(**{**CONFIG["mlp"], **extra})
    raise ValueError(f"unknown model: {name}")


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy.

    logits:  (B, T, vocab_size)
    targets: (B, T), -1 at the padding, which must not count.
    """
    loss = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-1)
    return loss



def train_epoch(model, loader, optimizer, device, tqdm_ = None) -> float:
    model.train()
    total, n = 0.0, 0
    for x, x_shifted, _, _ in loader:
        if tqdm_:
            tqdm_.set_description(f"train step {n + 1}/{len(loader)}")
            tqdm_.refresh()
        x, x_shifted = x.to(device), x_shifted.to(device)

        # steps: reset gradients, forward pass, compute loss, backward pass, clip gradients, optimizer step
        optimizer.zero_grad()
        forward = model(x)
        loss = lm_loss(forward, x_shifted)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # to make training more stable
        optimizer.step()
        total += loss.item()
        n += 1
    return total / n


@torch.no_grad()
def evaluate(model, loader, device, tqdm_ = None) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, x_shifted, _, _ in loader:
        if tqdm_:
            tqdm_.set_description(f"val step {n + 1}/{len(loader)}")
            tqdm_.refresh()
        x, x_shifted = x.to(device), x_shifted.to(device)
        
        # steps: forward pass, compute loss
        forward = model(x)
        loss = lm_loss(forward, x_shifted)
        total += loss.item()
        n += 1
    return total / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="transformer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=None, help="override the model's dropout rate")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit", action="store_true", help="train on a single batch")
    parser.add_argument("--wandb", action="store_true", help="log to wandb")
    parser.set_defaults(**CONFIG["train"])
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    device = get_device()
    _, _, _, train_loader, val_loader, _ = get_loaders(args.batch_size)
    if args.overfit:
        train_loader = [next(iter(train_loader))]
        val_loader = train_loader

    model = build_model(args.model, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"{args.model} on {device}: {sum(p.numel() for p in model.parameters()):,} parameters")

    logger = None
    if args.wandb:
        logger = wandb.init(
            entity="leonard-sima",
            project="autoaggressive-guidance",
            config=args.__dict__,
        )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val, history = float("inf"), []
    tqdm_ = tqdm(range(1, args.epochs + 1), desc="", leave=True, unit="epoch", ncols=80, colour="green")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, tqdm_)
        val_loss = evaluate(model, val_loader, device, tqdm_)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        val_ppl = math.exp(val_loss)
        history[-1]["val_ppl"] = val_ppl

        if val_loss < best_val:
            best_val = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }

            torch.save(checkpoint, f'{CKPT_DIR}/{args.model}.pt')

        if logger:
            logger.log({"val_ppl": val_ppl}, commit=False)
            logger.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch, "best_val_loss": best_val})

        # losses per epoch, for the learning curves in the report
        (CKPT_DIR / f"{args.model}_history.json").write_text(json.dumps(history, indent=1))
        tqdm_.update(1)
    tqdm_.close()
    if logger:
        logger.finish()

if __name__ == "__main__":
    main()
