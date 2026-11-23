"""Evaluate predicted family membership for generated sequences.

Results are saved as a table so figures can be created without the checkpoints.

Run from the repo root:
    uv run python eval_membership.py
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from eval_classifier import load_model
from protgen.data import AA, FAMILIES, total_tokens
from train_classifier import get_device

CKPT_DIR = Path("results/checkpoints")
SEQ_DIR = Path("results/sequences")
OUT_DIR = Path("results/classifier")

MAX_LEN = 129  # BOS + 127 residues + EOS
BOS = total_tokens.index("<BOS>")
EOS = total_tokens.index("<EOS>")
VOCAB = {residue: total_tokens.index(residue) for residue in AA}


def encode(sequences):
    """Encode sequences and return the position whose state has seen the complete sequence."""
    x = torch.zeros(len(sequences), MAX_LEN, dtype=torch.long)
    x[:, 0] = BOS
    positions = []

    for row, sequence in enumerate(sequences):
        ids = [VOCAB[residue] for residue in sequence][: MAX_LEN - 1]
        x[row, 1 : len(ids) + 1] = torch.tensor(ids)

        if len(ids) + 2 <= MAX_LEN:
            x[row, len(ids) + 1] = EOS
            positions.append(len(ids) + 1)
        else:
            # This sequence never emitted EOS, so its last residue is the only state that saw it all
            positions.append(len(ids))

    return x, torch.tensor(positions)


@torch.no_grad()
def predict(model, sequences, device, batch_size):
    x, positions = encode(sequences)
    out = []

    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size].to(device)
        at = positions[start : start + batch_size].to(device)
        logits = model(batch)[torch.arange(len(batch), device=device), at]
        out.append(F.softmax(logits, dim=-1).cpu())

    return torch.cat(out)


def settings(samples: pd.DataFrame) -> pd.Series:
    """Return the guidance setting stored in the sample file."""
    for column in ("setting", "lmbda"):
        if column in samples.columns:
            return samples[column].astype(str)

    return pd.Series("", index=samples.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=str(CKPT_DIR / "transformer.pt"))
    parser.add_argument("--head", default=str(CKPT_DIR / "classifier.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    model = load_model(Path(args.backbone), Path(args.head), device)
    columns = [f"p_{family}" for family in FAMILIES]
    tables = []

    for path in sorted(SEQ_DIR.glob("*.csv")):
        samples = pd.read_csv(path)
        sequences = samples["sequence"].fillna("").astype(str)

        unknown = set("".join(sequences)) - set(AA)
        if unknown:
            raise SystemExit(f"{path.name} has residues outside A: {sorted(unknown)}")

        # Empty sequences cannot be classified
        kept = sequences.str.len() > 0
        if not kept.all():
            print(f"{path.name}: dropped {(~kept).sum()} empty sequences")

        probabilities = predict(model, list(sequences[kept]), device, args.batch_size)
        table = pd.DataFrame(probabilities.numpy(), columns=columns)
        table.insert(0, "set", path.stem)
        table.insert(1, "setting", settings(samples)[kept].to_numpy())
        tables.append(table)

        print(f"{path.stem}: {len(table)} sequences")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(OUT_DIR / "membership.csv", index=False)
    print(f"wrote {OUT_DIR / 'membership.csv'}")


if __name__ == "__main__":
    main()
