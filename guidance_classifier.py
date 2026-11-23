"""Generate sequences guided towards a target family.

Run from the repo root:
    uv run python guidance_classifier.py --family AcyP --lmbda 1.0
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from eval_classifier import load_model
from protgen.data import AA, FAMILIES, total_tokens
from train_classifier import get_device

CKPT_DIR = Path("results/checkpoints")
OUT_DIR = Path("results/sequences")

BOS = total_tokens.index("<BOS>")
EOS = total_tokens.index("<EOS>")
RESIDUES = torch.tensor([total_tokens.index(residue) for residue in AA])

# Allow EOS, but exclude PAD and BOS from generation
CANDIDATES = torch.cat([RESIDUES, torch.tensor([EOS])])


@torch.no_grad()
def guided_step(model, prefix, family, lmbda):
    """Compute next-token scores with classifier guidance.

    log p(a | prefix) + lmbda * log p(family | prefix, a)
    """
    candidates = CANDIDATES.to(prefix.device)
    log_p = F.log_softmax(model.backbone(prefix)[:, -1, candidates], dim=-1)

    # The classifier scores the state after the candidate token, so every candidate needs its own
    # copy of the prefix with that token appended
    batch_size, n_candidates = log_p.shape
    candidate_prefixes = prefix.repeat_interleave(n_candidates, dim=0)
    candidate_prefixes = torch.cat(
        [candidate_prefixes, candidates.repeat(batch_size)[:, None]], dim=1
    )

    family_log_p = F.log_softmax(model(candidate_prefixes)[:, -1], dim=-1)[:, family]
    return log_p + lmbda * family_log_p.view(batch_size, n_candidates)


@torch.no_grad()
def check_lambda_zero(model, device):
    """Check that zero guidance gives the original model distribution.

    Says nothing about the candidate order, since multiplying by zero hides that too.
    """
    candidates, residues = CANDIDATES.to(device), RESIDUES.to(device)

    for step in (1, 7, 40):
        prefix = residues[torch.randint(len(AA), (4, step), device=device)]
        prefix[:, 0] = BOS
        plain = F.log_softmax(model.backbone(prefix)[:, -1, candidates], dim=-1)

        for family in range(len(FAMILIES)):
            guided = guided_step(model, prefix, family, 0.0)
            assert torch.allclose(guided, plain, atol=1e-6), (
                f"lmbda 0 differs at step {step}"
            )


@torch.no_grad()
def check_candidate_scores(model, device):
    """Check that the score in column j belongs to the prefix extended by candidate j.

    Scoring every candidate in one batch is the step that can silently transpose them.
    """
    candidates, residues = CANDIDATES.to(device), RESIDUES.to(device)
    prefix = residues[torch.randint(len(AA), (3, 9), device=device)]
    prefix[:, 0] = BOS

    plain = F.log_softmax(model.backbone(prefix)[:, -1, candidates], dim=-1)
    guided = guided_step(model, prefix, family=0, lmbda=1.0)

    for column, candidate in enumerate(candidates):
        with_candidate = torch.cat(
            [prefix, candidate.repeat(len(prefix))[:, None]], dim=1
        )
        expected = F.log_softmax(model(with_candidate)[:, -1], dim=-1)[:, 0]
        assert torch.allclose(
            guided[:, column] - plain[:, column], expected, atol=1e-4
        ), f"candidate {column} is scored out of order"


@torch.no_grad()
def generate(model, family, lmbda, n_sequences, max_length, device):
    candidates = CANDIDATES.to(device)
    x = torch.zeros(n_sequences, max_length, dtype=torch.long, device=device)
    x[:, 0] = BOS
    finished = torch.zeros(n_sequences, dtype=torch.bool, device=device)

    for step in range(1, max_length):
        log_p = guided_step(model, x[:, :step], family, lmbda)
        picked = candidates[torch.multinomial(log_p.exp(), 1)[:, 0]]

        # Keep finished sequences padded while the remaining ones continue
        x[:, step] = torch.where(finished, torch.zeros_like(picked), picked)
        finished |= picked == EOS

        if finished.all():
            break

    return x


def decode(x):
    # Decode without loading a dataset
    residues = {total_tokens.index(residue): residue for residue in AA}
    return ["".join(residues[int(t)] for t in row if int(t) in residues) for row in x]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=str(CKPT_DIR / "transformer.pt"))
    parser.add_argument("--head", default=str(CKPT_DIR / "classifier.pt"))
    parser.add_argument("--family", default="AcyP", choices=FAMILIES)
    parser.add_argument("--lmbda", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=129)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    model = load_model(Path(args.backbone), Path(args.head), device)
    check_lambda_zero(model, device)
    check_candidate_scores(model, device)

    family = FAMILIES.index(args.family)
    started = time.perf_counter()
    x = generate(model, family, args.lmbda, args.n, args.max_length, device)
    elapsed = time.perf_counter() - started

    sequences = decode(x)
    print(
        f"{args.family} lmbda {args.lmbda:g} on {device}: "
        f"{len(sequences)} sequences in {elapsed:.1f} s"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"guided_{args.family}_lmbda{args.lmbda:g}.csv"

    pd.DataFrame(
        {
            "sequence": sequences,
            "target_family": args.family,
            "lmbda": args.lmbda,
        }
    ).to_csv(out, index=False)

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
