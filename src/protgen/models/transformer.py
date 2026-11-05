"""Decoder-only (causal) transformer for autoregressive protein sequence modelling.

Input:  token ids x of shape (B, T)
Output: logits of shape (B, T, vocab_size); position i predicts token i + 1.
"""
import math

import torch
import torch.nn as nn


def causal_mask(T: int, device=None) -> torch.Tensor:
    """Boolean mask of shape (1, 1, T, T): True where attention is allowed.

    Position i may look at positions j <= i only.
    """
    # -> defines `mask`, bool, shape (1, 1, T, T)
    mask = torch.tril(torch.ones(T, T), diagonal=0).bool()
    mask = mask.unsqueeze(0).unsqueeze(0).to(device=device)
    return mask


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)  # each have shape (B, T, C)

        # Split into heads.
        # -> q, k, v each of shape (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product scores, causal mask, softmax over the key axis, dropout
        mask = causal_mask(T, x.device)
        
        # att shape (B, n_heads, T, T), rows sum to 1
        att = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        att = torch.masked_fill(att, ~mask, float("-inf")).softmax(dim=-1)        
        att = self.attn_dropout(att)

        # out shape (B, T, C)
        out = att @ v
        out = out.transpose(1, 2)
        out = out.reshape([B, T, C])
        return self.resid_dropout(self.proj(out))


class Block(nn.Module):
    # block = attention + MLP, with residual connections and layer norm
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 23,
        max_len: int = 129,
        d_model: int = 128, # embedding dimension
        n_layers: int = 4, # number of transformer blocks
        n_heads: int = 4, # number of attention heads
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.max_len, f"sequence length {T} exceeds max_len {self.max_len}"

        # token embedding + positional embedding, then dropout
        # h shape (B, T, d_model)
        embed = self.tok_emb(x)
        pos_embed = self.pos_emb(torch.arange(0, T).to(device=x.device))
        h = self.drop(embed + pos_embed)

        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        if return_hidden: # hidden states for the classifier head in phase 2
            return h
        return self.head(h)
