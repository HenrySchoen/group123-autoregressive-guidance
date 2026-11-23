"""Family classifier using a frozen transformer backbone."""

import torch
import torch.nn as nn

from protgen.data import FAMILIES, IGNORE_INDEX


def valid_positions(x: torch.Tensor) -> torch.Tensor:
    """True for valid positions, where position i represents the first i residues."""
    mask = x != 0  # PAD is id 0
    mask[:, 0] = False  # skip BOS
    return mask


def family_targets(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Assign the sequence family label to each valid position."""
    targets = torch.full_like(x, IGNORE_INDEX)
    mask = valid_positions(x)
    targets[mask] = y[:, None].expand_as(x)[mask]
    return targets


class FamilyClassifier(nn.Module):
    def __init__(
        self, backbone: nn.Module, d_model: int | None = None, n_families: int = len(FAMILIES)
    ):
        super().__init__()
        self.backbone = backbone
        if d_model is None:
            d_model = backbone.tok_emb.embedding_dim

        # Only train the classifier head (freeze the backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Linear(d_model, n_families)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # keep the frozen backbone in eval mode
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The causal backbone computes all prefix states in one pass
        with torch.no_grad():
            hidden = self.backbone(x, return_hidden=True)
        return self.head(hidden)
