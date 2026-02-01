"""
Position Classifier Model - GPT for predicting absolute position

This model takes token sequences and predicts the absolute position (0 to block_size-1)
at each position. Used for studying how transformers encode position without
positional embeddings.

Based on model_nope.py but with:
- Position classification head instead of LM head
- Cross-entropy loss over positions
- No weight tying with embeddings
"""

import math
import inspect
from dataclasses import dataclass
from typing import Literal

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


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            att = att.masked_fill(causal_mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config):
        super().__init__()
        mlp_ratio = getattr(config, "mlp_ratio", 4)
        hidden_dim = mlp_ratio * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(self, config):
        super().__init__()
        NormClass = RMSNorm if config.norm_type == "rmsnorm" else LayerNorm
        self.ln_1 = NormClass(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.use_ln2 = getattr(
            config, "use_ln2", True
        )  # Default True for backward compat
        if self.use_ln2:
            self.ln_2 = NormClass(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        if self.use_ln2:
            x = x + self.mlp(self.ln_2(x))
        else:
            x = x + self.mlp(x)  # No normalization before MLP
        return x


@dataclass
class GPTPositionClassifierConfig:
    """Configuration for Position Classifier model."""

    block_size: int = 128  # Sequence length = number of position classes
    vocab_size: int = 50304
    n_layer: int = 1
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    use_positional_embedding: bool = False  # NoPE mode
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    use_regression: bool = (
        False  # If True, use MSE loss for regression instead of classification
    )
    compute_lm_loss: bool = False  # If True, also compute LM perplexity
    use_ln2: bool = (
        True  # If False, skip ln_2 (LayerNorm after attention) in transformer blocks
    )
    mlp_expansion_ratio: int = 4  # Expansion ratio for MLP hidden dimension (default 4)


class GPTPositionClassifier(nn.Module):
    """
    GPT model for position classification.

    Instead of predicting next token, predicts absolute position (0 to block_size-1)
    at each position in the sequence.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        NormClass = RMSNorm if config.norm_type == "rmsnorm" else LayerNorm

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=NormClass(config.n_embd, bias=config.bias),
            )
        )

        # Optionally add positional embeddings
        if config.use_positional_embedding:
            self.transformer["wpe"] = nn.Embedding(config.block_size, config.n_embd)

        # Position prediction head
        if config.use_regression:
            # Regression: predict continuous position value
            self.pos_head = nn.Linear(config.n_embd, 1, bias=True)
        else:
            # Classification: predict position class 0 to block_size-1
            self.pos_head = nn.Linear(config.n_embd, config.block_size, bias=False)

        # Optional LM head for perplexity evaluation
        if config.compute_lm_loss:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            # Weight tying with token embeddings
            self.lm_head.weight = self.transformer.wte.weight

        # Initialize weights
        self.apply(self._init_weights)
        self._init_attention_xavier()

        # Scaled initialization for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        print(f"\n{'=' * 60}")
        print(f"Position Classifier Model Initialized")
        print(f"{'=' * 60}")
        print(f"  Positional Embedding: {config.use_positional_embedding}")
        print(f"  Normalization Type:   {config.norm_type}")
        print(f"  Layers:               {config.n_layer}")
        print(f"  Heads:                {config.n_head}")
        print(f"  Embedding Dim:        {config.n_embd}")
        print(f"  Block Size:           {config.block_size}")
        print(f"  Vocab Size:           {config.vocab_size}")
        print(f"  Position Classes:     {config.block_size}")
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
        """Apply Xavier initialization to Q, K, V projections."""
        for block in self.transformer.h:
            nn.init.xavier_uniform_(block.attn.c_attn.weight)
            if block.attn.c_attn.bias is not None:
                nn.init.zeros_(block.attn.c_attn.bias)

    def forward(self, idx, targets=None, lm_targets=None):
        """
        Forward pass for position prediction.

        Args:
            idx: Input token indices [batch_size, seq_len]
            targets: Position labels [batch_size, seq_len], values 0 to seq_len-1
                     For regression, these are converted to float
            lm_targets: LM targets for perplexity computation [batch_size, seq_len]
                       (next token prediction targets)

        Returns:
            logits/preds: Position predictions
                - Classification: [batch_size, seq_len, block_size] logits
                - Regression: [batch_size, seq_len] predicted positions
            pos_loss: Cross-entropy (classification) or MSE (regression) loss
            lm_loss: LM cross-entropy loss (None if compute_lm_loss is False)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Sequence length {t} > block size {self.config.block_size}"
        )

        # Token embeddings
        tok_emb = self.transformer.wte(idx)

        # Add positional embeddings only if enabled
        if self.config.use_positional_embedding and "wpe" in self.transformer:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # LM prediction (optional, for perplexity)
        lm_loss = None
        if hasattr(self, "lm_head") and lm_targets is not None:
            lm_logits = self.lm_head(x)
            lm_loss = F.cross_entropy(
                lm_logits.view(-1, self.config.vocab_size), lm_targets.view(-1)
            )

        # Position prediction
        if self.config.use_regression:
            # Regression: output single value per position
            preds = self.pos_head(x).squeeze(-1)  # [batch, seq_len]

            if targets is not None:
                # Normalize targets to [0, 1] range for better training
                targets_normalized = targets.float() / (self.config.block_size - 1)
                preds_normalized = torch.sigmoid(preds)  # Ensure output in [0, 1]
                pos_loss = F.mse_loss(preds_normalized, targets_normalized)
            else:
                pos_loss = None

            return preds, pos_loss, lm_loss
        else:
            # Classification
            logits = self.pos_head(x)  # [batch, seq_len, block_size]

            if targets is not None:
                pos_loss = F.cross_entropy(
                    logits.view(-1, self.config.block_size), targets.view(-1)
                )
            else:
                pos_loss = None

            return logits, pos_loss, lm_loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure AdamW optimizer with weight decay."""
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
            f"Decayed params: {len(decay_params)} tensors, {num_decay_params:,} params"
        )
        print(
            f"Non-decayed params: {len(nodecay_params)} tensors, {num_nodecay_params:,} params"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"Using fused AdamW: {use_fused}")

        return optimizer

    def freeze_transformer(self):
        """
        Freeze all parameters except the position classification head.
        Used for probing experiments where only the head is trained.
        """
        for name, param in self.named_parameters():
            if "pos_head" not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Frozen transformer. Trainable: {trainable:,} / {total:,} params")
        print(f"Only pos_head is trainable: {self.pos_head.weight.shape}")

    def freeze_attention_only(self):
        """
        Freeze embeddings and attention layers.
        Keep MLP layers and position head trainable.

        Frozen:
        - transformer.wte (token embeddings)
        - transformer.wpe (positional embeddings, if present)
        - transformer.h.{i}.ln_1 (pre-attention norm)
        - transformer.h.{i}.attn.* (all attention parameters)

        Trainable:
        - transformer.h.{i}.ln_2 (pre-MLP norm)
        - transformer.h.{i}.mlp.* (all MLP parameters)
        - transformer.ln_f (final norm)
        - pos_head (position prediction head)
        - lm_head (if present, for perplexity evaluation)
        """
        frozen_count = 0
        trainable_count = 0

        for name, param in self.named_parameters():
            # Freeze embeddings
            if "wte" in name or "wpe" in name:
                param.requires_grad = False
                frozen_count += param.numel()
            # Freeze attention and pre-attention norm
            elif "attn" in name or "ln_1" in name:
                param.requires_grad = False
                frozen_count += param.numel()
            # Keep MLPs, ln_2, ln_f, pos_head, and lm_head trainable
            else:
                param.requires_grad = True
                trainable_count += param.numel()

        total = sum(p.numel() for p in self.parameters())
        print(f"\n*** FREEZE ATTENTION ONLY MODE ***")
        print(f"Frozen: {frozen_count:,} params (embeddings + attention)")
        print(f"Trainable: {trainable_count:,} params (MLPs + norms + heads)")
        print(f"Total: {total:,} params")

    def freeze_until_first_mlp(self):
        """
        Freeze everything up until (but not including) the first MLP.

        Frozen:
        - transformer.wte (token embeddings)
        - transformer.wpe (positional embeddings, if present)
        - transformer.h.0.ln_1 (pre-attention norm of first block)
        - transformer.h.0.attn.* (attention of first block)
        - transformer.h.0.ln_2 (pre-MLP norm of first block)

        Trainable:
        - transformer.h.0.mlp.* (first MLP - this is where training starts)
        - transformer.h.{1+}.* (all subsequent blocks fully trainable)
        - transformer.ln_f (final norm)
        - pos_head (position prediction head)
        - lm_head (if present, for perplexity evaluation)
        """
        frozen_count = 0
        trainable_count = 0

        for name, param in self.named_parameters():
            # Freeze embeddings
            if "wte" in name or "wpe" in name:
                param.requires_grad = False
                frozen_count += param.numel()
            # Freeze first block's attention and norms (but not MLP)
            elif "transformer.h.0.ln_1" in name:
                param.requires_grad = False
                frozen_count += param.numel()
            elif "transformer.h.0.attn" in name:
                param.requires_grad = False
                frozen_count += param.numel()
            elif "transformer.h.0.ln_2" in name:
                param.requires_grad = False
                frozen_count += param.numel()
            # Everything else is trainable (first MLP, all other blocks, final norm, heads)
            else:
                param.requires_grad = True
                trainable_count += param.numel()

        total = sum(p.numel() for p in self.parameters())
        print(f"\n*** FREEZE UNTIL FIRST MLP MODE ***")
        print(f"Frozen: {frozen_count:,} params (embeddings + block0 attention/norms)")
        print(f"Trainable: {trainable_count:,} params (block0 MLP + blocks 1+ + heads)")
        print(f"Total: {total:,} params")
