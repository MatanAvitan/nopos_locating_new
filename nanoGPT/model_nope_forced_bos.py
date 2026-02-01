"""
NoPE GPT Model with Forced BOS Mechanism

This model extends the standard NoPE GPT to "hard-code" position encoding:
- Block 0, Head 0: Frozen to uniform causal attention (prefix averaging)
- Block 1, Head 0: Frozen to attend only to position 0 (BOS head)

All other parameters are trainable as usual.
"""

import math
from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, ndim, bias=False, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        if self.bias is not None:
            return self.weight * x_normed + self.bias
        return self.weight * x_normed


class ForcedBOSCausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional frozen heads.

    Supports:
    - freeze_uniform_head: Freeze specified head to uniform causal attention
    - freeze_bos_head: Freeze specified head to attend only to position 0
    """

    def __init__(self, config, block_idx: int = 0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_idx = block_idx

        # Frozen head configuration
        self.freeze_uniform_head_idx = getattr(config, "freeze_uniform_head_idx", None)
        self.freeze_bos_head_idx = getattr(config, "freeze_bos_head_idx", None)
        self.uniform_head_block = getattr(config, "uniform_head_block", 0)
        self.bos_head_block = getattr(config, "bos_head_block", 1)

        # Determine if this block has frozen heads
        self.has_frozen_uniform = (
            block_idx == self.uniform_head_block
            and self.freeze_uniform_head_idx is not None
        )
        self.has_frozen_bos = (
            block_idx == self.bos_head_block and self.freeze_bos_head_idx is not None
        )

        self.log_attention_stats = config.log_attention_stats
        self.last_attention_weights = None

    def _get_uniform_causal_attention(
        self, T: int, device: torch.device, dtype: torch.dtype
    ):
        """Generate uniform causal attention pattern: each position attends equally to all previous."""
        # Create lower triangular matrix of ones
        mask = torch.tril(torch.ones(T, T, device=device, dtype=dtype))
        # Normalize each row to sum to 1
        attn = mask / mask.sum(dim=-1, keepdim=True)
        return attn  # [T, T]

    def _get_bos_attention(self, T: int, device: torch.device, dtype: torch.dtype):
        """Generate BOS attention pattern: all positions attend only to position 0."""
        attn = torch.zeros(T, T, device=device, dtype=dtype)
        attn[:, 0] = 1.0  # All queries attend to key position 0
        return attn  # [T, T]

    def forward(self, x):
        B, T, C = x.size()
        device = x.device
        dtype = x.dtype

        # Calculate Q, K, V for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Always compute attention manually when we have frozen heads
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        att = att.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        # Replace frozen heads with fixed patterns (clone to avoid in-place modification)
        if self.has_frozen_uniform or self.has_frozen_bos:
            att = att.clone()  # Clone to avoid in-place modification breaking autograd

        if self.has_frozen_uniform:
            uniform_attn = self._get_uniform_causal_attention(T, device, dtype)
            att[:, self.freeze_uniform_head_idx, :, :] = uniform_attn.unsqueeze(0)

        if self.has_frozen_bos:
            bos_attn = self._get_bos_attention(T, device, dtype)
            att[:, self.freeze_bos_head_idx, :, :] = bos_attn.unsqueeze(0)

        if self.log_attention_stats and self.training:
            self.last_attention_weights = att[-1].detach().cpu()

        att = self.attn_dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(self, config, block_idx: int = 0):
        super().__init__()
        NormClass = RMSNorm if config.norm_type == "rmsnorm" else LayerNorm
        self.ln_1 = NormClass(config.n_embd, bias=config.bias)
        self.attn = ForcedBOSCausalSelfAttention(config, block_idx=block_idx)
        self.ln_2 = NormClass(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfigForcedBOS:
    """Configuration for NoPE GPT model with forced BOS mechanism."""

    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    use_positional_embedding: bool = False
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    log_attention_stats: bool = False

    # Forced BOS mechanism configuration
    freeze_uniform_head_idx: Optional[int] = 0  # Head index for uniform attention
    freeze_bos_head_idx: Optional[int] = 0  # Head index for BOS attention
    uniform_head_block: int = 0  # Block index for uniform head
    bos_head_block: int = 1  # Block index for BOS head


class GPTForcedBOS(nn.Module):
    """
    GPT Language Model with Forced BOS Mechanism.

    Hard-codes position encoding by freezing:
    - Block 0, Head 0: Uniform causal attention (prefix averaging)
    - Block 1, Head 0: BOS attention (attend only to position 0)
    """

    def __init__(self, config: GPTConfigForcedBOS):
        super().__init__()
        self.config = config

        NormClass = RMSNorm if config.norm_type == "rmsnorm" else LayerNorm

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [Block(config, block_idx=i) for i in range(config.n_layer)]
                ),
                ln_f=NormClass(config.n_embd, bias=config.bias),
            )
        )

        if config.use_positional_embedding:
            self.transformer["wpe"] = nn.Embedding(config.block_size, config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        self._init_attention_xavier()

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        print(f"\n{'=' * 60}")
        print(f"NoPE GPT with Forced BOS Mechanism Initialized")
        print(f"{'=' * 60}")
        print(f"  Layers:               {config.n_layer}")
        print(f"  Heads:                {config.n_head}")
        print(f"  Embedding Dim:        {config.n_embd}")
        print(f"  Block Size:           {config.block_size}")
        print(
            f"  Uniform Head:         Block {config.uniform_head_block}, Head {config.freeze_uniform_head_idx}"
        )
        print(
            f"  BOS Head:             Block {config.bos_head_block}, Head {config.freeze_bos_head_idx}"
        )
        print(f"  Parameters:           {self.get_num_params() / 1e6:.2f}M")
        print(f"{'=' * 60}\n")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and "wpe" in self.transformer:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_attention_xavier(self):
        for block in self.transformer.h:
            nn.init.xavier_uniform_(block.attn.c_attn.weight)
            if block.attn.c_attn.bias is not None:
                nn.init.zeros_(block.attn.c_attn.bias)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size

        tok_emb = self.transformer.wte(idx)
        if "wpe" in self.transformer:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
