"""Build the report figures.

Reads each table where its run left it and writes PNGs to report/figures/.

Run from the repo root, after eval_membership.py:
    uv run python figures.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from protgen.data import AA, FAMILIES
from protgen.metrics.composition import composition, family_means, family_metrics

CKPT_DIR = Path("results/checkpoints")
CLASSIFIER_DIR = Path("results/classifier")
EVAL_DIR = Path("results/eval")
SEQ_DIR = Path("results/sequences")
OUT_DIR = Path("report/figures")

TARGET = "AcyP"

# Full page width of the template, so the font size here is the size on the page
PAGE = 7.0

# Models rather than guidance strengths
BASELINES = {
    "mlp": "MLP",
    "guided_AcyP_lmbda0": "Transformer",
    "control_real": "real proteins",
    "control_shuffled": "shuffled",
}

# Guidance strength per scored set
CLASSIFIER_SETS = {
    "guided_AcyP_lmbda0": 0.0,
    "guided_AcyP_lmbda0.5": 0.5,
    "guided_AcyP_lmbda1": 1.0,
    "guided_AcyP_lmbda2": 2.0,
    "guided_AcyP_lmbda5": 5.0,
}
CFG_SETS = {
    "cfg_null_label": -1.0,
    "cfg_w0": 0.0,
    "cfg_w1": 1.0,
    "cfg_w2": 2.0,
    "cfg_w4": 4.0,
    "cfg_w8": 8.0,
}

# Below a hundred sequences per family the rates are noise
MIN_SEQUENCES = 300


def save(fig, out: Path, name: str, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()

    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"wrote {out / name}.png")


def eval_metrics() -> pd.DataFrame:
    """Per-sequence metrics for every scored set."""
    tables = []

    for path in sorted(EVAL_DIR.glob("*/eval.csv")):
        table = pd.read_csv(path)
        table["set"] = path.parent.name
        tables.append(table)

    if not tables:
        raise SystemExit(f"no eval.csv under {EVAL_DIR}")

    return pd.concat(tables, ignore_index=True)


def history(name: str) -> pd.DataFrame:
    return pd.DataFrame(json.loads((CKPT_DIR / f"{name}_history.json").read_text()))


def learning_curves(out: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(PAGE, 3.0), sharey=True,
                                      layout="constrained")

    for name, label in (("mlp", "MLP"), ("transformer", "Transformer")):
        table = history(name)
        line, = left.plot(table["epoch"], table["val_loss"], label=f"{label} validation")
        left.plot(table["epoch"], table["train_loss"], "--", color=line.get_color(),
                  label=f"{label} training")

    left.set_title("unconditional models")
    left.set_xlabel("epoch")
    left.set_ylabel("cross entropy (nats per token)")

    cfg = history("cfg")
    right.plot(cfg["epoch"], cfg["val_loss"], color="C2", label="with family label")
    right.plot(cfg["epoch"], cfg["val_null_loss"], color="C4", label="null label")
    right.set_title("classifier-free model")
    right.set_xlabel("epoch")

    # One legend for both panels, there is no room for it inside either
    lines = left.get_lines() + right.get_lines()
    fig.legend(lines, [line.get_label() for line in lines], loc="outside lower center", ncol=3)

    save(fig, out, "learning_curves", tight=False)


def generation_quality(metrics: pd.DataFrame, out: Path) -> None:
    present = [name for name in BASELINES if name in set(metrics["set"])]
    labels = [BASELINES[name] for name in present]
    fig, (left, right) = plt.subplots(1, 2, figsize=(PAGE, 2.8))

    for axes, column, label in (
        (left, "mean_plddt", "pLDDT"),
        (right, "nearest_identity", "identity to nearest training sequence"),
    ):
        values = [metrics.loc[metrics["set"] == name, column].dropna() for name in present]
        axes.boxplot(values, tick_labels=labels)
        axes.set_ylabel(label)
        axes.tick_params(axis="x", rotation=15)

    save(fig, out, "generation_quality")


def prefix_curve(out: Path) -> None:
    table = pd.read_csv(CLASSIFIER_DIR / "prefix_accuracy.csv")
    table = table[table["n_sequences"] >= MIN_SEQUENCES]

    fig, axes = plt.subplots(figsize=(PAGE, 2.8))

    for family in FAMILIES:
        axes.plot(table["prefix_length"], table[family], label=family)

    axes.plot(table["prefix_length"], table["accuracy"], "k--", label="all families")
    axes.axhline(1 / len(FAMILIES), color="grey", lw=0.8, label="chance")
    axes.set_xlabel("prefix length (residues)")
    axes.set_ylabel("accuracy")
    axes.legend(ncol=5, loc="lower center")

    save(fig, out, "prefix_accuracy")


def strength(setting: str) -> float:
    """Map the null label run to -1 so it sorts before w = 0."""
    return -1.0 if setting == "null_label" else float(setting.removeprefix("w="))


def add_measurements(curve: pd.DataFrame, metrics: pd.DataFrame, membership: pd.DataFrame,
                     sets: dict, strengths: pd.Series) -> pd.DataFrame:
    """Attach pLDDT and predicted membership to a curve indexed by guidance strength."""
    folded = metrics[metrics["set"].isin(sets)].copy()
    folded["strength"] = folded["set"].map(sets)
    curve["plddt"] = folded.groupby("strength")["mean_plddt"].mean()

    scored = membership.copy()
    scored["strength"] = strengths
    curve["p_target"] = scored.groupby("strength")[f"p_{TARGET}"].mean()

    return curve


def classifier_curve(metrics, membership, means) -> pd.DataFrame:
    rows = []

    for path in sorted(SEQ_DIR.glob(f"guided_{TARGET}_lmbda*.csv")):
        samples = pd.read_csv(path)
        distances = family_metrics(samples["sequence"], means)[f"l1_{TARGET}"]
        rows.append({"strength": float(samples["lmbda"].iloc[0]), "l1": distances.mean()})

    curve = pd.DataFrame(rows).set_index("strength").sort_index()
    scored = membership[membership["set"].str.startswith(f"guided_{TARGET}_lmbda")]

    return add_measurements(curve, metrics, scored, CLASSIFIER_SETS,
                            scored["setting"].astype(float))


def cfg_curve(metrics, membership, means) -> pd.DataFrame:
    samples = pd.read_csv(SEQ_DIR / "guided_cfg.csv")
    sequences = samples["sequence"].fillna("").astype(str)
    kept = sequences.str.len() > 0

    distances = family_metrics(sequences[kept], means)[f"l1_{TARGET}"]
    curve = pd.DataFrame({"l1": distances, "strength": samples["setting"][kept].map(strength)})
    curve = curve.groupby("strength").mean()

    scored = membership[membership["set"] == "guided_cfg"]

    return add_measurements(curve, metrics, scored, CFG_SETS, scored["setting"].map(strength))


def tradeoff(metrics, membership, means, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(PAGE, 4.6), sharex="col")
    panels = (
        (classifier_curve(metrics, membership, means), "classifier guidance", r"$\lambda$"),
        (cfg_curve(metrics, membership, means), "classifier-free guidance", r"$w$"),
    )

    for column, (curve, title, xlabel) in enumerate(panels):
        top, bottom = axes[0][column], axes[1][column]

        top.plot(curve.index, curve["l1"], marker="o", color="C0")
        top.tick_params(axis="y", colors="C0")
        top.set_title(title)

        predictor = top.twinx()
        predictor.plot(curve.index, curve["p_target"], marker="s", color="C3")
        predictor.tick_params(axis="y", colors="C3")
        predictor.set_ylim(0, 1)

        bottom.plot(curve.index, curve["plddt"], marker="o", color="C2")
        bottom.set_xlabel(f"guidance strength {xlabel}")

        if column == 0:
            top.set_ylabel(f"L1 distance to {TARGET} (points)", color="C0")
            bottom.set_ylabel("mean pLDDT")
        else:
            predictor.set_ylabel(f"predicted p({TARGET})", color="C3")

    # The null label run has no w, so it gets its own tick rather than a number
    axes[1][1].set_xticks([-1, 0, 2, 4, 8], ["null", "0", "2", "4", "8"])

    save(fig, out, "tradeoff")


def composition_profile(means: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(figsize=(PAGE, 2.8))
    positions = range(len(AA))
    width = 0.27

    for offset, lmbda in ((-width, 0), (0.0, 1)):
        samples = pd.read_csv(SEQ_DIR / f"guided_{TARGET}_lmbda{lmbda}.csv")
        fractions = composition(samples["sequence"]).mean() * 100
        axes.bar([p + offset for p in positions], fractions, width,
                 label=rf"generated, $\lambda$ = {lmbda}")

    axes.bar([p + width for p in positions], means.loc[TARGET] * 100, width,
             label=f"real {TARGET}")
    axes.set_xticks(list(positions), list(AA))
    axes.set_xlabel("amino acid")
    axes.set_ylabel("abundance (percent)")
    axes.legend()

    save(fig, out, "composition_profile")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--reference", default="data/train.csv")
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 9, "savefig.dpi": 200})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics = eval_metrics()
    membership = pd.read_csv(CLASSIFIER_DIR / "membership.csv")
    means = family_means(args.reference)

    learning_curves(out)
    generation_quality(metrics, out)
    prefix_curve(out)
    tradeoff(metrics, membership, means, out)
    composition_profile(means, out)


if __name__ == "__main__":
    main()
