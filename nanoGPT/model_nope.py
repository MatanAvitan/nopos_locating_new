"""
NoPE GPT Model - GPT without Positional Embeddings
Supports LayerNorm and RMSNorm variants for studying positional encoding emergence.

Key features:
- Configurable positional embeddings (disabled by default for NoPE)
- Choice of LayerNorm or RMSNorm normalization
- Xavier initialization for attention Q, K, V matrices
- Attention statistics logging (entropy, uniformity) during training
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
    """
    Root Mean Square Layer Normalization.
    Unlike LayerNorm, RMSNorm does NOT center the activations (no mean subtraction).
    This is important for studying how position information flows differently.

    RMSNorm(x) = x / RMS(x) * γ
    where RMS(x) = sqrt(mean(x^2))
    """

    def __init__(self, ndim, bias=False, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        # RMSNorm typically doesn't use bias, but include for compatibility
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        if self.bias is not None:
            return self.weight * x_normed + self.bias
        return self.weight * x_normed


class BatchNorm1dForTransformer(nn.Module):
    """
    BatchNorm wrapper for transformer sequences.
    Standard BatchNorm1d expects (B, C, T), but transformers use (B, T, C).
    This wrapper handles the transpose.

    Key insight: BatchNorm normalizes across the BATCH dimension, which may
    preserve population-level positional statistics that LayerNorm destroys.
    This is used to test whether replacing LN2 with BatchNorm helps NoPE
    models maintain position information through population statistics.
    """

    def __init__(self, ndim, bias=True, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(ndim, eps=eps, momentum=momentum, affine=True)
        # Note: bias parameter ignored - BatchNorm1d always has affine params

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T) for BatchNorm -> (B, T, C)
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional attention statistics logging.

    When log_attention_stats=True, computes and stores:
    - Attention entropy per head (measure of attention spread)
    - Attention uniformity per head (correlation with 1/(i+1) variance pattern)
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Q, K, V projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # For logging attention statistics
        self.log_attention_stats = config.log_attention_stats
        self.last_attention_weights = None  # Store for analysis
        self.last_attention_entropy = None
        self.last_attention_uniformity = None

        # Flash attention - disable when logging stats to get attention weights
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()

        # Calculate Q, K, V for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if self.flash and not self.log_attention_stats:
            # Use efficient flash attention when not logging stats
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # Manual attention computation to capture weights
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Causal mask
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            att = att.masked_fill(causal_mask, float("-inf"))
            att = F.softmax(att, dim=-1)

            # Log attention statistics if enabled
            if self.log_attention_stats and self.training:
                self._compute_attention_stats(att)

            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    def _compute_attention_stats(self, att):
        """
        Compute attention statistics for logging during training.

        Metrics:
        - Entropy: H = -sum(p * log(p)), higher = more spread out attention
        - Uniformity: Correlation of attention variance with 1/(i+1) pattern
          (uniform causal attention at position i has variance 1/(i+1))
        """
        with torch.no_grad():
            B, H, T, _ = att.shape

            # Store last batch attention weights for potential analysis
            self.last_attention_weights = att[-1].detach().cpu()  # [n_head, T, T]

            # Compute entropy per head per position
            # H = -sum(p * log(p)), where p is attention probability
            att_clamped = att.clamp(min=1e-10)
            entropy = -torch.sum(
                att_clamped * torch.log(att_clamped), dim=-1
            )  # [B, H, T]
            # Average over batch and positions to get per-head entropy
            self.last_attention_entropy = entropy.mean(dim=(0, 2))  # [H]

            # Compute uniformity: correlation with expected 1/(i+1) variance pattern
            # For uniform attention over positions 0..i, variance = 1/(i+1)
            positions = torch.arange(1, T + 1, device=att.device, dtype=att.dtype)
            expected_uniform_var = 1.0 / positions  # [T]

            # Actual variance of attention weights per position
            actual_var = att.var(dim=-1)  # [B, H, T]
            actual_var_mean = actual_var.mean(dim=0)  # [H, T]

            # Compute correlation for each head
            uniformity_scores = []
            for h in range(H):
                head_var = actual_var_mean[h]  # [T]
                # Pearson correlation
                x = head_var - head_var.mean()
                y = expected_uniform_var - expected_uniform_var.mean()
                corr = (x * y).sum() / (torch.sqrt((x**2).sum() * (y**2).sum()) + 1e-10)
                uniformity_scores.append(corr.item())

            self.last_attention_uniformity = uniformity_scores


class MLP(nn.Module):
    """Feed-forward network with GELU activation and 4x expansion."""

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
    Transformer block with pre-norm architecture.

    x -> LN/RMSNorm -> Attention -> + -> LN/RMSNorm/BN -> MLP -> +
         |__________________________|    |______________________|
                  residual                       residual

    If skip_ln2=True, the second LayerNorm is skipped (for ablation study):
    x -> LN/RMSNorm -> Attention -> + -> MLP -> +
         |__________________________|    |_____|
                  residual              residual

    If use_batchnorm_ln2=True, ln_2 uses BatchNorm instead of LayerNorm/RMSNorm.
    This tests whether BatchNorm preserves population-level positional statistics.
    """

    def __init__(self, config):
        super().__init__()
        NormClass = RMSNorm if config.norm_type == "rmsnorm" else LayerNorm
        self.ln_1 = NormClass(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.skip_ln2 = config.skip_ln2
        self.use_batchnorm_ln2 = getattr(config, "use_batchnorm_ln2", False)
        if not self.skip_ln2:
            if self.use_batchnorm_ln2:
                self.ln_2 = BatchNorm1dForTransformer(config.n_embd, bias=config.bias)
            else:
                self.ln_2 = NormClass(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        if self.skip_ln2:
            x = x + self.mlp(x)
        else:
            x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    """
    Configuration for NoPE GPT model.

    Key NoPE-specific options:
    - use_positional_embedding: Set to False for NoPE experiments
    - norm_type: 'layernorm' or 'rmsnorm' for studying normalization effects
    - log_attention_stats: Enable to track attention patterns during training
    - skip_ln2: Skip second LayerNorm (for ablation studying LN2's effect on position encoding)
    - use_batchnorm_ln2: Use BatchNorm instead of LayerNorm for ln_2 (preserves population stats)
    """

    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    # NoPE-specific options
    use_positional_embedding: bool = False  # Set to False for NoPE
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    log_attention_stats: bool = (
        False  # Log attention entropy/uniformity during training
    )
    skip_ln2: bool = False  # Skip second LayerNorm for ablation
    use_batchnorm_ln2: bool = False  # Use BatchNorm instead of LayerNorm for ln_2


class GPT(nn.Module):
    """
    GPT Language Model without Positional Embeddings (NoPE).

    This model is designed for studying how transformers encode position
    implicitly without explicit positional embeddings.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Select normalization class
        NormClass = RMSNorm if config.norm_type == "rmsnorm" else LayerNorm

        # Build transformer
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=NormClass(config.n_embd, bias=config.bias),
            )
        )

        # Conditionally add positional embeddings
        if config.use_positional_embedding:
            self.transformer["wpe"] = nn.Embedding(config.block_size, config.n_embd)

        # Language model head (tied with token embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)

        # Apply Xavier initialization to attention Q, K, V (critical for near-uniform attention)
        self._init_attention_xavier()

        # Scaled initialization for residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # Report model configuration
        print(f"\n{'=' * 60}")
        print(f"NoPE GPT Model Initialized")
        print(f"{'=' * 60}")
        print(f"  Positional Embedding: {config.use_positional_embedding}")
        print(f"  Normalization Type:   {config.norm_type}")
        print(f"  Layers:               {config.n_layer}")
        print(f"  Heads:                {config.n_head}")
        print(f"  Embedding Dim:        {config.n_embd}")
        print(f"  Block Size:           {config.block_size}")
        print(f"  Vocab Size:           {config.vocab_size}")
        print(f"  Attention Stats:      {config.log_attention_stats}")
        print(f"  Skip LN2:             {config.skip_ln2}")
        print(f"  Use BatchNorm LN2:    {config.use_batchnorm_ln2}")
        print(f"  Parameters:           {self.get_num_params() / 1e6:.2f}M")
        print(f"{'=' * 60}\n")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), position embeddings are subtracted.
        Token embeddings are included due to weight tying with lm_head.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and "wpe" in self.transformer:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Standard weight initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_attention_xavier(self):
        """
        Apply Xavier initialization to Q, K, V projections.

        This is critical for achieving near-uniform attention at initialization,
        which is important for the positional encoding mechanism described in
        the paper "How Causal Transformers Encode Position Without Positional Embeddings".

        Xavier init ensures attention scores have appropriate variance for
        the softmax to produce near-uniform distributions initially.
        """
        for block in self.transformer.h:
            # c_attn contains Q, K, V concatenated: [n_embd, 3*n_embd]
            nn.init.xavier_uniform_(block.attn.c_attn.weight)
            if block.attn.c_attn.bias is not None:
                nn.init.zeros_(block.attn.c_attn.bias)

    def forward(self, idx, targets=None):
        """
        Forward pass of the NoPE GPT model.

        Args:
            idx: Input token indices [batch_size, seq_len]
            targets: Target token indices for loss computation (optional)

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            loss: Cross-entropy loss if targets provided, else None
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Sequence length {t} > block size {self.config.block_size}"
        )

        # Token embeddings
        tok_emb = self.transformer.wte(idx)  # [b, t, n_embd]

        # Add positional embeddings only if enabled
        if self.config.use_positional_embedding and "wpe" in self.transformer:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)  # [t, n_embd]
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            # NoPE mode: no positional embedding added
            x = self.transformer.drop(tok_emb)

        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Compute logits and optionally loss
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Inference optimization: only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def get_attention_stats(self):
        """
        Retrieve attention statistics from all layers.

        Returns dict with:
        - layer_{i}_entropy: Mean attention entropy per head [n_head]
        - layer_{i}_uniformity: Uniformity correlation per head [n_head]
        """
        stats = {}
        for i, block in enumerate(self.transformer.h):
            attn = block.attn
            if attn.last_attention_entropy is not None:
                stats[f"layer_{i}_entropy"] = attn.last_attention_entropy
                stats[f"layer_{i}_uniformity"] = attn.last_attention_uniformity
        return stats

    def crop_block_size(self, block_size):
        """Reduce block size for finetuning on shorter sequences."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if "wpe" in self.transformer:
            self.transformer.wpe.weight = nn.Parameter(
                self.transformer.wpe.weight[:block_size]
            )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias") and block.attn.bias is not None:
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure AdamW optimizer with weight decay applied only to 2D+ params.
        Embeddings and matmul weights get decay; biases and layernorms don't.
        """
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

        # Use fused AdamW if available (faster on CUDA)
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model FLOPs utilization (MFU) relative to A100 bfloat16 peak."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak FLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens autoregressively.

        Args:
            idx: Conditioning sequence [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = no change)
            top_k: If set, only sample from top k tokens

        Returns:
            Extended sequence [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop sequence if it exceeds block size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # Get logits for last position
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
