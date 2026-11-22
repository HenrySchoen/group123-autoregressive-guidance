"""Sample protein sequences from a trained model.

Run from the repo root:
    uv run python sample.py --model mlp --num 256
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from protgen.data import total_tokens
from train import CKPT_DIR, CONFIG, build_model, get_device

BOS = total_tokens.index("<BOS>")
EOS = total_tokens.index("<EOS>")
MAX_LEN = 129


@torch.no_grad()
def sample(model, num, temperature=1.0, device="cpu"):
    model.eval()
    # every sequence starts with <BOS>
    x = torch.full((num, 1), BOS, dtype=torch.long, device=device)
    done = torch.zeros(num, dtype=torch.bool, device=device)

    while x.size(1) < MAX_LEN and not done.all():
        logits = model(x)[:, -1, :]  # only need the prediction for the last position
        probs = torch.softmax(logits / temperature, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_tok], dim=1)
        done = done | (next_tok.squeeze(1) == EOS)
    return x


def decode(tokens):
    # token ids -> amino acid string, stop at <EOS> and skip the special tokens
    seq = ""
    for t in tokens[1:]:
        if t == EOS:
            break
        if t >= 3:
            seq += total_tokens[t]
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--num", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.set_defaults(**CONFIG["sample"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()

    model = build_model(args.model).to(device)
    ckpt = torch.load(CKPT_DIR / f"{args.model}.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    tokens = sample(model, args.num, args.temperature, device)
    seqs = [decode(row) for row in tokens.tolist()]

    out = Path(args.out or f"results/sequences/{args.model}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sequence": seqs}).to_csv(out, index=False)
    print(f"wrote {len(seqs)} sequences to {out}")


if __name__ == "__main__":
    main()