import torch
import torch.nn as nn

from protgen.data import FAMILIES
from protgen.models.transformer import Transformer

NULL_LABEL = len(FAMILIES)  # 3, the empty label for classifier-free guidance


class ConditionalTransformer(Transformer):
    # same transformer, but the family label is added to the input embedding

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d_model = self.tok_emb.embedding_dim
        self.label_emb = nn.Embedding(len(FAMILIES) + 1, d_model)
        self._init_weights(self.label_emb)

    def forward(self, x, label=None, return_hidden=False):
        B, T = x.shape
        if label is None:  # no label given means the null label
            label = torch.full((B,), NULL_LABEL, device=x.device)

        pos = torch.arange(0, T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos) + self.label_emb(label).unsqueeze(1)
        h = self.drop(h)

        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        if return_hidden:
            return h
        return self.head(h)