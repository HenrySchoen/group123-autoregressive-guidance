import torch.nn as nn
import torch.nn.functional as F

from protgen.data import total_tokens


class MLP(nn.Module):
    # MLP from Bengio et al. 2003, predicts the next token from the last 3 tokens

    def __init__(self, vocab_size=len(total_tokens), context=3, embed_dim=32,
                 hidden_dim=256, dropout=0.1):
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(context * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x):
        # x: (B, L) -> logits: (B, L, vocab_size)
        w = self.windows(x)          # (B, L, 3)
        e = self.embed(w)            # (B, L, 3, embed_dim)
        e = e.flatten(start_dim=2)   # concat the 3 embeddings
        return self.net(e)

    def windows(self, x):
        # pad with 0s on the left so the first positions also get 3 tokens
        padded = F.pad(x, (self.context - 1, 0), value=0)
        return padded.unfold(1, self.context, 1)