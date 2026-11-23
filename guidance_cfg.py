"""Classifier-free guidance sampling from the label dropout model.

Run from the repo root:
    uv run python guidance_cfg.py
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from protgen.data import FAMILIES
from protgen.models.transformer_cfg import ConditionalTransformer, NULL_LABEL
from sample import BOS, EOS, MAX_LEN, decode
from train import CKPT_DIR, CONFIG, get_device


@torch.no_grad()
def cfg_sample(model, num, label, w, temperature=1.0, device="cpu"):
    """Sample with classifier-free guidance. label=None gives the null label run."""
    model.eval()
    x = torch.full((num, 1), BOS, dtype=torch.long, device=device)
    done = torch.zeros(num, dtype=torch.bool, device=device)

    while x.size(1) < MAX_LEN and not done.all():
        if label is None:
            logits = model(x, None)[:, -1, :]
        else:
            y = torch.full((num,), label, dtype=torch.long, device=device)
            logits_cond = model(x, y)[:, -1, :]
            logits_null = model(x, None)[:, -1, :]
            # push away from the unconditional prediction
            logits = (1 + w) * logits_cond - w * logits_null

        probs = torch.softmax(logits / temperature, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_tok], dim=1)
        done = done | (next_tok.squeeze(1) == EOS)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--family", default=None)
    parser.add_argument("--out", default="results/sequences/guided_cfg.csv")
    parser.set_defaults(family=CONFIG["cfg"]["target_family"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()

    model = ConditionalTransformer(**CONFIG["transformer"]).to(device)
    ckpt = torch.load(CKPT_DIR / "cfg.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    label = FAMILIES.index(args.family)
    rows = []

    # the unguided reference: the model with the null label
    tokens = cfg_sample(model, args.num, None, 0.0, args.temperature, device)
    rows += [{"sequence": decode(t), "setting": "null_label"} for t in tokens.tolist()]

    for w in CONFIG["cfg"]["w_values"]:
        tokens = cfg_sample(model, args.num, label, w, args.temperature, device)
        rows += [{"sequence": decode(t), "setting": f"w={w}"} for t in tokens.tolist()]
        print(f"sampled w={w}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} sequences to {out}")


if __name__ == "__main__":
    main()