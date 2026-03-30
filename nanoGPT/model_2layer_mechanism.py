"""
2-Layer NoPE Model for Mechanism Analysis

This model is specifically designed for the mechanism dissection experiments
outlined in the 2-layer NoPE position encoding study. It provides:

1. Exactly 2 pre-norm transformer blocks
2. Comprehensive tap points for all intermediate activations
3. Flexible freezing regimes for ablation studies (R0-R4)
4. Attention weight capture for analysis
5. Hooks for all component outputs

Architecture (pre-norm):
Block 1:
    x_i^1 = LN1(r_i^0)          # Pre-attention norm
    a_i^1 = Attn1(x^1)_i        # Block 1 attention output
    r_i^1 = r_i^0 + a_i^1       # Post-attention residual
    m_i^1 = MLP1(LN2(r_i^1))    # Block 1 MLP output
    r_i^1_out = r_i^1 + m_i^1   # Block 1 output

Block 2:
    x_i^2 = LN3(r_i^1_out)      # Pre-attention norm
    a_i^2 = Attn2(x^2)_i        # Block 2 attention output
    r_i^2 = r_i^1_out + a_i^2   # Post-attention residual
    m_i^2 = MLP2(LN4(r_i^2))    # Block 2 MLP output
    r_i^2_out = r_i^2 + m_i^2   # Block 2 output (final)
"""

import math
import inspect
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, List, Tuple

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
    """
    Multi-head causal self-attention with attention weight capture.
    """

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
        self.head_dim = config.n_embd // config.n_head

        # Attention weight capture (only when using manual attention)
        self.last_attention_weights = None

        # Flash attention support for long contexts
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.use_flash = getattr(config, "use_flash", False)

    def forward(self, x, return_attn_weights=False):
        B, T, C = x.size()

        # Calculate Q, K, V for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.use_flash and self.flash and not return_attn_weights:
            # Use flash attention for O(n) memory — no weight capture
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            self.last_attention_weights = None
        else:
            # Manual attention to capture weights (O(n²) memory)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            att = att.masked_fill(causal_mask, float("-inf"))
            att = F.softmax(att, dim=-1)

            # Store attention weights
            self.last_attention_weights = att.detach()

            att = self.attn_dropout(att)
            y = att @ v

        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        if return_attn_weights:
            return y, self.last_attention_weights
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
    """
    Transformer block with pre-norm architecture and activation taps.
    """

    def __init__(self, config, block_idx):
        super().__init__()
        NormClass = RMSNorm if config.norm_type == "rmsnorm" else LayerNorm
        self.ln_1 = NormClass(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = NormClass(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.block_idx = block_idx

        # Tap points for intermediate activations
        self.last_ln1_out = None  # x_i = LN(r_i^{in})
        self.last_attn_out = None  # a_i = Attn(x)
        self.last_post_attn = None  # r_i^1 = r_i^{in} + a_i
        self.last_ln2_out = None  # LN(r_i^1)
        self.last_mlp_out = None  # m_i = MLP(LN(r_i^1))
        self.last_block_out = None  # r_i^{out} = r_i^1 + m_i

    def forward(self, x, capture_taps=True):
        # Pre-attention LayerNorm
        ln1_out = self.ln_1(x)
        if capture_taps:
            self.last_ln1_out = ln1_out.detach()

        # Attention
        attn_out = self.attn(ln1_out)
        if capture_taps:
            self.last_attn_out = attn_out.detach()

        # Post-attention residual
        post_attn = x + attn_out
        if capture_taps:
            self.last_post_attn = post_attn.detach()

        # Pre-MLP LayerNorm
        ln2_out = self.ln_2(post_attn)
        if capture_taps:
            self.last_ln2_out = ln2_out.detach()

        # MLP
        mlp_out = self.mlp(ln2_out)
        if capture_taps:
            self.last_mlp_out = mlp_out.detach()

        # Post-MLP residual
        block_out = post_attn + mlp_out
        if capture_taps:
            self.last_block_out = block_out.detach()

        return block_out

    def forward_no_mlp(self, x, capture_taps=True):
        """Forward pass that skips the MLP (post-attn residual only)."""
        ln1_out = self.ln_1(x)
        if capture_taps:
            self.last_ln1_out = ln1_out.detach()

        attn_out = self.attn(ln1_out)
        if capture_taps:
            self.last_attn_out = attn_out.detach()

        post_attn = x + attn_out
        if capture_taps:
            self.last_post_attn = post_attn.detach()

        ln2_out = self.ln_2(post_attn)
        if capture_taps:
            self.last_ln2_out = ln2_out.detach()

        if capture_taps:
            self.last_mlp_out = None

        if capture_taps:
            self.last_block_out = post_attn.detach()

        return post_attn


@dataclass
class TwoLayerMechanismConfig:
    """
    Configuration for 2-Layer Mechanism Analysis Model.

    Follows the notation from the experiment spec:
    - d = n_embd (embedding dimension)
    - n_heads = 12 per spec
    - d_head = d / n_heads = 64 per spec
    - MLP hidden = 4 * d
    """

    block_size: int = 128  # Sequence length L
    vocab_size: int = 50304  # GPT-2 vocab size
    n_embd: int = 768  # d=768 per spec
    n_head: int = 12  # 12 heads per spec
    dropout: float = 0.0  # No dropout for clean analysis
    bias: bool = True  # Include biases
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    use_regression: bool = True  # MSE loss for position regression
    use_flash: bool = False  # Use flash attention (O(n) memory, no weight capture)


class TwoLayerMechanismModel(nn.Module):
    """
    2-Layer Pre-Norm Causal Transformer for Mechanism Analysis.

    This model is designed for position regression experiments to understand
    how NoPE transformers encode position through attention averaging.

    Supports 5 freezing regimes:
    - R0: Full training (all parameters trainable)
    - R1: Block2-only (freeze Emb, Attn1, MLP1)
    - R2: Attn2-only (freeze Emb, Attn1, MLP1, MLP2)
    - R3: MLP2-only (freeze Emb, Attn1, MLP1, Attn2)
    - R4: Head-only probe (freeze all transformer params)
    """

    def __init__(self, config: TwoLayerMechanismConfig):
        super().__init__()
        self.config = config

        # Token embeddings (no positional embeddings - NoPE)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Exactly 2 transformer blocks
        self.block1 = Block(config, block_idx=0)
        self.block2 = Block(config, block_idx=1)

        # Final layer norm
        NormClass = RMSNorm if config.norm_type == "rmsnorm" else LayerNorm
        self.ln_f = NormClass(config.n_embd, bias=config.bias)

        # Position regression head
        if config.use_regression:
            # Linear head for position regression
            self.pos_head = nn.Linear(config.n_embd, 1, bias=True)
        else:
            # Classification head
            self.pos_head = nn.Linear(config.n_embd, config.block_size, bias=False)

        # Optional: 2-layer MLP probe (for R4 variants)
        self.mlp_probe = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, 1 if config.use_regression else config.block_size),
        )
        self.use_mlp_probe = False  # Toggle via method
        self.use_post_attn_head = False

        # Initialize weights
        self.apply(self._init_weights)
        self._init_attention_xavier()

        # Report configuration
        print(f"\n{'=' * 60}")
        print(f"2-Layer Mechanism Analysis Model Initialized")
        print(f"{'=' * 60}")
        print(f"  Embedding Dim:        {config.n_embd}")
        print(f"  Heads:                {config.n_head}")
        print(f"  Head Dim:             {config.n_embd // config.n_head}")
        print(f"  Block Size:           {config.block_size}")
        print(f"  Normalization:        {config.norm_type}")
        print(f"  Regression Mode:      {config.use_regression}")
        print(f"  Parameters:           {self.get_num_params() / 1e6:.2f}M")
        print(f"{'=' * 60}\n")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_attention_xavier(self):
        """Apply Xavier init to Q, K, V for near-uniform attention at init."""
        for block in [self.block1, self.block2]:
            nn.init.xavier_uniform_(block.attn.c_attn.weight)
            if block.attn.c_attn.bias is not None:
                nn.init.zeros_(block.attn.c_attn.bias)

    def forward(self, idx, targets=None, capture_taps=True):
        """
        Forward pass with optional intermediate activation capture.

        Args:
            idx: Token indices [B, T]
            targets: Position targets [B, T] (values 0 to T-1)
            capture_taps: Whether to store intermediate activations

        Returns:
            preds: Position predictions
            loss: MSE or CE loss if targets provided
        """
        B, T = idx.size()

        # Token embeddings (r_i^0 = e_i)
        tok_emb = self.wte(idx)  # [B, T, d]
        x = self.drop(tok_emb)

        # Store embeddings for analysis
        if capture_taps:
            self.last_embeddings = tok_emb.detach()

        # Block 1
        x = self.block1(x, capture_taps=capture_taps)

        # Block 2
        if self.use_post_attn_head:
            x = self.block2.forward_no_mlp(x, capture_taps=capture_taps)
        else:
            x = self.block2(x, capture_taps=capture_taps)

        # Final layer norm
        x = self.ln_f(x)

        # Position prediction
        if self.use_mlp_probe:
            output = self.mlp_probe(x)
        else:
            output = self.pos_head(x)

        # Compute loss
        loss = None
        if targets is not None:
            if self.config.use_regression:
                preds = output.squeeze(-1)  # [B, T]
                loss = F.mse_loss(preds, targets.float())
            else:
                loss = F.cross_entropy(
                    output.view(-1, self.config.block_size), targets.view(-1)
                )

        return output, loss

    def set_post_attn_head(self, enable: bool = True) -> None:
        """Apply head on block2 post-attention residual, skipping MLP2."""
        self.use_post_attn_head = enable

    def get_all_taps(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieve all intermediate activations from the last forward pass.

        Returns dictionary with keys:
        - 'embeddings': e_i token embeddings
        - 'block1_ln1': x_i^1 = LN1(r_i^0)
        - 'block1_attn': a_i^1 = Attn1(x^1)
        - 'block1_post_attn': r_i^1 = r_i^0 + a_i^1
        - 'block1_ln2': LN2(r_i^1)
        - 'block1_mlp': m_i^1 = MLP1(LN2(r_i^1))
        - 'block1_out': r_i^1_out = r_i^1 + m_i^1
        - 'block2_ln1': x_i^2 = LN3(r_i^1_out)
        - 'block2_attn': a_i^2 = Attn2(x^2)
        - 'block2_post_attn': r_i^2 = r_i^1_out + a_i^2
        - 'block2_ln2': LN4(r_i^2)
        - 'block2_mlp': m_i^2 = MLP2(LN4(r_i^2))
        - 'block2_out': r_i^2_out = r_i^2 + m_i^2
        """
        return {
            "embeddings": self.last_embeddings,
            "block1_ln1": self.block1.last_ln1_out,
            "block1_attn": self.block1.last_attn_out,
            "block1_post_attn": self.block1.last_post_attn,
            "block1_ln2": self.block1.last_ln2_out,
            "block1_mlp": self.block1.last_mlp_out,
            "block1_out": self.block1.last_block_out,
            "block2_ln1": self.block2.last_ln1_out,
            "block2_attn": self.block2.last_attn_out,
            "block2_post_attn": self.block2.last_post_attn,
            "block2_ln2": self.block2.last_ln2_out,
            "block2_mlp": self.block2.last_mlp_out,
            "block2_out": self.block2.last_block_out,
        }

    def get_attention_weights(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get attention weights from both blocks.

        Returns:
            attn1: Block 1 attention weights [B, n_head, T, T]
            attn2: Block 2 attention weights [B, n_head, T, T]
        """
        return (
            self.block1.attn.last_attention_weights,
            self.block2.attn.last_attention_weights,
        )

    # =========================================================================
    # Freezing Regimes for Experiment 1
    # =========================================================================

    def apply_regime_R0(self):
        """R0: Full training baseline - all parameters trainable."""
        for param in self.parameters():
            param.requires_grad = True
        self._report_trainable("R0: Full Training")

    def apply_regime_R1(self):
        """R1: Block2-only - freeze Emb, Attn1, MLP1, train Attn2, MLP2, Head."""
        for param in self.parameters():
            param.requires_grad = False

        # Train Block 2 (ln_1, attn, ln_2, mlp)
        for param in self.block2.parameters():
            param.requires_grad = True

        # Train final LN and head
        for param in self.ln_f.parameters():
            param.requires_grad = True
        for param in self.pos_head.parameters():
            param.requires_grad = True
        for param in self.mlp_probe.parameters():
            param.requires_grad = True

        self._report_trainable("R1: Block2-only")

    def apply_regime_R2(self):
        """R2: Attn2-only - freeze Emb, Attn1, MLP1, MLP2, train Attn2, Head."""
        for param in self.parameters():
            param.requires_grad = False

        # Train only Attn2 (ln_1 and attn of block2)
        for param in self.block2.ln_1.parameters():
            param.requires_grad = True
        for param in self.block2.attn.parameters():
            param.requires_grad = True

        # Train final LN and head
        for param in self.ln_f.parameters():
            param.requires_grad = True
        for param in self.pos_head.parameters():
            param.requires_grad = True
        for param in self.mlp_probe.parameters():
            param.requires_grad = True

        self._report_trainable("R2: Attn2-only")

    def apply_regime_R2_attn_head_only(self):
        """R2: Attn2-only (strict) - train only Block2.Attn and Pos Head."""
        for param in self.parameters():
            param.requires_grad = False

        for param in self.block2.attn.parameters():
            param.requires_grad = True
        for param in self.pos_head.parameters():
            param.requires_grad = True

        self.use_mlp_probe = False
        self._report_trainable("R2: Attn2-only (strict)")

    def apply_regime_R3(self):
        """R3: MLP2-only - freeze Emb, Attn1, MLP1, Attn2, train MLP2, Head."""
        for param in self.parameters():
            param.requires_grad = False

        # Train only MLP2 (ln_2 and mlp of block2)
        for param in self.block2.ln_2.parameters():
            param.requires_grad = True
        for param in self.block2.mlp.parameters():
            param.requires_grad = True

        # Train final LN and head
        for param in self.ln_f.parameters():
            param.requires_grad = True
        for param in self.pos_head.parameters():
            param.requires_grad = True
        for param in self.mlp_probe.parameters():
            param.requires_grad = True

        self._report_trainable("R3: MLP2-only")

    def apply_regime_R4(self, use_mlp_probe=False):
        """R4: Head-only probe - freeze all transformer params, train only head."""
        for param in self.parameters():
            param.requires_grad = False

        # Train only the head
        if use_mlp_probe:
            self.use_mlp_probe = True
            for param in self.mlp_probe.parameters():
                param.requires_grad = True
        else:
            self.use_mlp_probe = False
            for param in self.pos_head.parameters():
                param.requires_grad = True

        probe_type = "MLP probe" if use_mlp_probe else "Linear head"
        self._report_trainable(f"R4: Head-only ({probe_type})")

    def _report_trainable(self, regime_name: str):
        """Report trainable parameter count."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        frozen = total - trainable

        print(f"\n{'=' * 60}")
        print(f"Regime: {regime_name}")
        print(f"{'=' * 60}")
        print(f"  Trainable:  {trainable:,} params ({100 * trainable / total:.1f}%)")
        print(f"  Frozen:     {frozen:,} params ({100 * frozen / total:.1f}%)")
        print(f"  Total:      {total:,} params")

        # Detailed breakdown
        param_groups = {
            "Embedding": self.wte,
            "Block1.LN1": self.block1.ln_1,
            "Block1.Attn": self.block1.attn,
            "Block1.LN2": self.block1.ln_2,
            "Block1.MLP": self.block1.mlp,
            "Block2.LN1": self.block2.ln_1,
            "Block2.Attn": self.block2.attn,
            "Block2.LN2": self.block2.ln_2,
            "Block2.MLP": self.block2.mlp,
            "Final LN": self.ln_f,
            "Pos Head": self.pos_head,
            "MLP Probe": self.mlp_probe,
        }

        print("\n  Component status:")
        for name, module in param_groups.items():
            n_params = sum(p.numel() for p in module.parameters())
            n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            status = "TRAIN" if n_train > 0 else "frozen"
            print(f"    {name:15s}: {status:6s} ({n_params:,} params)")
        print(f"{'=' * 60}\n")

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure AdamW optimizer with weight decay."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        if not param_dict:
            raise ValueError("No trainable parameters! Apply a regime first.")

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        print(
            f"Optimizer: {len(decay_params)} weight decay, {len(nodecay_params)} no decay"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused if fused_available else False,
        )

        return optimizer


# =========================================================================
# Analysis Utilities
# =========================================================================


def compute_position_metrics(
    preds: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute comprehensive position regression metrics.

    Args:
        preds: Predicted positions [B, T] or [B*T]
        targets: Target positions [B, T] or [B*T]

    Returns:
        Dictionary with MAE, R^2, per-position MAE, calibration error
    """
    preds = preds.float().flatten()
    targets = targets.float().flatten()

    # MAE
    mae = (preds - targets).abs().mean().item()

    # R^2
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0

    # Per-position MAE (reshape if possible)
    B_T = len(preds)

    return {
        "mae": mae,
        "r2": r2,
        "rmse": math.sqrt(((preds - targets) ** 2).mean().item()),
    }


def compute_per_position_mae(
    preds: torch.Tensor, targets: torch.Tensor, block_size: int
) -> torch.Tensor:
    """
    Compute MAE for each position separately.

    Args:
        preds: [B, T]
        targets: [B, T]
        block_size: Sequence length

    Returns:
        per_pos_mae: [T] tensor of MAE per position
    """
    B, T = preds.shape
    per_pos_mae = torch.zeros(T)
    for pos in range(T):
        per_pos_mae[pos] = (preds[:, pos] - targets[:, pos]).abs().mean()
    return per_pos_mae
