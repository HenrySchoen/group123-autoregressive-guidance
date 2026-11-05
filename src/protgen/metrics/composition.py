"""Evaluate family membership using amino acid composition.

Independent of the classifier, so guided samples can be evaluated without the model that steered them.
"""

import pandas as pd

from protgen.data import AA, FAMILIES

AROMATIC = "FWY"
POSITIVE = "KR"
NEGATIVE = "DE"


def composition(sequences) -> pd.DataFrame:
    """Calculate amino acid fractions for each sequence."""
    sequences = pd.Series(sequences)
    counts = pd.DataFrame(
        [[sequence.count(residue) for residue in AA] for sequence in sequences],
        columns=list(AA),
        index=sequences.index,
    )
    return counts.div(counts.sum(axis=1), axis=0)


def family_means(path: str) -> pd.DataFrame:
    """Mean composition per family from a CSV with sequence and family columns."""
    data = pd.read_csv(path)
    means = composition(data["sequence"]).groupby(data["family"]).mean()
    return means.reindex(list(FAMILIES))


def family_metrics(sequences, means: pd.DataFrame) -> pd.DataFrame:
    """Compare sequences to family means using L1 distance and residue markers."""
    comp = composition(sequences)
    table = pd.DataFrame(index=comp.index)

    for family in FAMILIES:
        # Express L1 distances in percentage points
        table[f"l1_{family}"] = (comp - means.loc[family]).abs().sum(axis=1) * 100

    table["aromatic"] = comp[list(AROMATIC)].sum(axis=1)
    table["ala"] = comp["A"]
    table["net_charge"] = comp[list(POSITIVE)].sum(axis=1) - comp[list(NEGATIVE)].sum(
        axis=1
    )
    return table
