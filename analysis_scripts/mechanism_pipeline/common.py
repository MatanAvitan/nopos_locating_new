"""Shared utilities for the canonical mechanism extraction pipeline (exps.md).

Loads the paper checkpoints (ATTN2-1H / FULL-12H), builds deterministic
disjoint sequence splits from OpenWebText val.bin, and runs forwards that
expose every quantity the mechanism analysis needs (per-head scores,
attention weights, OV images, residual-stream taps).
"""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
NANOGPT = REPO_ROOT / "nanoGPT"
RESULTS_ROOT = REPO_ROOT / "results" / "mechanism"

CHECKPOINTS = {
    "attn2_1h": NANOGPT / "out-mechanism-R2-1024/R2/t72g9e8p/best_ckpt.pt",
    "full12h": NANOGPT / "out-mechanism-R0-1024/R0/nuacla0w/best_ckpt.pt",
    # ATTN2-FULL (exps.md P1): fully trained 2-layer single-head. Wandb run
    # ids vary, so these are glob patterns resolved in load_model.
    "attn2_full": NANOGPT / "out-mechanism-ATTN2FULL-1024-s1/R0/*/best_ckpt.pt",
    "attn2_full_lr3": NANOGPT / "out-mechanism-ATTN2FULL-1024-lr3e4/R0/*/best_ckpt.pt",
    "attn2_full_s42fail": NANOGPT / "out-mechanism-ATTN2FULL-1024/R0/*/best_ckpt.pt",
    "attn2_full_seed1": NANOGPT / "out-mechanism-ATTN2FULL-128-seed1/R0/*/best_ckpt.pt",
    "attn2_full_seed2": NANOGPT / "out-mechanism-ATTN2FULL-128-seed2/R0/*/best_ckpt.pt",
    "attn2_full_seed3": NANOGPT / "out-mechanism-ATTN2FULL-128-seed3/R0/*/best_ckpt.pt",
    "attn2_full_seed4": NANOGPT / "out-mechanism-ATTN2FULL-128-seed4/R0/*/best_ckpt.pt",
}


def resolve_checkpoint(model_name: str) -> Path:
    path = CHECKPOINTS[model_name]
    if "*" in str(path):
        matches = sorted(path.parent.parent.glob(f"*/{path.name}"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"{model_name}: expected exactly one match for {path}, "
                f"got {matches}")
        path = matches[0]
    return path

VAL_BIN = NANOGPT / "data/openwebtext/val.bin"
BOS_TOKEN_ID = 50256


def add_nanogpt_to_path():
    p = str(NANOGPT)
    if p not in sys.path:
        sys.path.insert(0, p)


def sha256_file(path, max_bytes=64 * 1024 * 1024):
    """Hash the first max_bytes of a file (checkpoints are large)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_model(model_name: str, device: str = "cuda", init_only: bool = False):
    """Load a paper checkpoint by short name ('attn2_1h' or 'full12h').

    These checkpoints store hyperparameters under ckpt['config'] (a plain
    dict), not 'model_args'. If init_only, return a freshly initialized
    model with the same architecture and seed instead of trained weights.
    """
    add_nanogpt_to_path()
    from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

    ckpt_path = resolve_checkpoint(model_name)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_cfg = TwoLayerMechanismConfig(
        block_size=cfg["block_size"],
        vocab_size=cfg["vocab_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        dropout=0.0,
        bias=cfg.get("bias", True),
        norm_type=cfg.get("norm_type", "layernorm"),
        use_regression=True,
        use_flash=False,
    )
    torch.manual_seed(cfg.get("seed", 42))
    model = TwoLayerMechanismModel(model_cfg)
    if not init_only:
        state = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in ckpt["model"].items()
        }
        model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    meta = {
        "model_name": model_name,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_hash": sha256_file(ckpt_path),
        "iter_num": ckpt.get("iter_num"),
        "best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
        "regime": cfg.get("regime"),
        "n_head": cfg["n_head"],
        "n_embd": cfg["n_embd"],
        "block_size_train": cfg["block_size"],
        "seed": cfg.get("seed", 42),
        "bos_token_id": cfg.get("bos_token_id", BOS_TOKEN_ID),
        "init_only": init_only,
    }
    return model, meta


# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------

SPLIT_SIZES = {"reference": 640, "calibration": 320, "evaluation": 1600}


def build_splits(context_length: int, seed: int = 1234,
                 sizes: Optional[Dict[str, int]] = None) -> Dict[str, np.ndarray]:
    """Deterministic, non-overlapping start offsets into val.bin.

    Each sequence is BOS + (context_length - 1) tokens starting at the
    offset. Offsets are drawn without replacement from a non-overlapping
    grid, then partitioned into the three disjoint splits.
    """
    sizes = sizes or SPLIT_SIZES
    data_len = os.path.getsize(VAL_BIN) // 2  # uint16
    tokens_per_seq = context_length - 1
    n_slots = data_len // tokens_per_seq
    total = sum(sizes.values())
    if total > n_slots:
        raise ValueError(
            f"Need {total} non-overlapping sequences of {tokens_per_seq} tokens, "
            f"val.bin only supports {n_slots}"
        )
    rng = np.random.default_rng(seed)
    slots = rng.permutation(n_slots)[:total]
    offsets = slots * tokens_per_seq
    splits, k = {}, 0
    for name in ["reference", "calibration", "evaluation"]:
        splits[name] = np.sort(offsets[k:k + sizes[name]])
        k += sizes[name]
    return splits


def batch_from_offsets(offsets: np.ndarray, context_length: int,
                       device: str) -> torch.Tensor:
    data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    seqs = []
    for off in offsets:
        toks = data[off: off + context_length - 1].astype(np.int64)
        seqs.append(np.concatenate([[BOS_TOKEN_ID], toks]))
    return torch.from_numpy(np.stack(seqs)).to(device)


def iter_batches(offsets: np.ndarray, context_length: int, batch_size: int,
                 device: str):
    for s in range(0, len(offsets), batch_size):
        chunk = offsets[s: s + batch_size]
        yield s, batch_from_offsets(chunk, context_length, device)


# ---------------------------------------------------------------------------
# Forward with full capture
# ---------------------------------------------------------------------------

@dataclass
class HeadWeights:
    """Per-head weight slices for one attention block."""
    W_q: torch.Tensor  # [H, dh, d]
    W_k: torch.Tensor
    W_v: torch.Tensor
    b_q: torch.Tensor  # [H, dh]
    b_k: torch.Tensor
    b_v: torch.Tensor
    W_o: torch.Tensor  # [H, d, dh]  (output projection slice per head)
    b_o: torch.Tensor  # [d]
    scale: float


def head_weights(attn) -> HeadWeights:
    d = attn.n_embd
    H = attn.n_head
    dh = d // H
    Wqkv = attn.c_attn.weight  # [3d, d]
    bqkv = attn.c_attn.bias if attn.c_attn.bias is not None else torch.zeros(
        3 * d, device=Wqkv.device, dtype=Wqkv.dtype)
    Wq, Wk, Wv = Wqkv[:d], Wqkv[d:2 * d], Wqkv[2 * d:]
    bq, bk, bv = bqkv[:d], bqkv[d:2 * d], bqkv[2 * d:]
    Wo = attn.c_proj.weight  # [d, d]
    bo = attn.c_proj.bias if attn.c_proj.bias is not None else torch.zeros(
        d, device=Wo.device, dtype=Wo.dtype)
    return HeadWeights(
        W_q=Wq.view(H, dh, d), W_k=Wk.view(H, dh, d), W_v=Wv.view(H, dh, d),
        b_q=bq.view(H, dh), b_k=bk.view(H, dh), b_v=bv.view(H, dh),
        W_o=Wo.view(d, H, dh).permute(1, 0, 2).contiguous(), b_o=bo,
        scale=1.0 / (dh ** 0.5),
    )


def attn_scores_weights(x: torch.Tensor, hw: HeadWeights,
                        dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scores and softmax weights per head. Returns (scores, weights),
    each [B, H, T, T] with -inf above the diagonal in scores."""
    q = torch.einsum("btd,hed->bhte", x.to(dtype), hw.W_q.to(dtype)) + hw.b_q.to(dtype)[None, :, None, :]
    k = torch.einsum("btd,hed->bhte", x.to(dtype), hw.W_k.to(dtype)) + hw.b_k.to(dtype)[None, :, None, :]
    scores = torch.einsum("bhte,bhse->bhts", q, k) * hw.scale
    T = x.shape[1]
    mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return scores, weights


def ov_images(x: torch.Tensor, hw: HeadWeights, dtype=torch.float32) -> torch.Tensor:
    """y^{(h)}_j = B_OV^{(h)} x_j = W_O^h W_V^h x_j, WITHOUT the value bias
    (manuscript convention: b_attn := W_O b_V + b_O absorbs the biases).

    Returns [B, H, T, d]. Exactness: since softmax weights sum to 1 per head,
    o = sum_h sum_j alpha_hj y_hj + b_attn.
    """
    v = torch.einsum("btd,hed->bhte", x.to(dtype), hw.W_v.to(dtype))
    y = torch.einsum("bhte,hde->bhtd", v, hw.W_o.to(dtype))
    return y


def attn_bias_vector(hw: HeadWeights, dtype=torch.float32) -> torch.Tensor:
    """b_attn := W_O b_V + b_O summed over heads. With softmax weights summing
    to 1 per head, each head contributes W_O^h b_v^h exactly once."""
    per_head = torch.einsum("he,hde->hd", hw.b_v.to(dtype), hw.W_o.to(dtype))
    return per_head.sum(0) + hw.b_o.to(dtype)


@torch.no_grad()
def forward_capture(model, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Full forward returning every tap plus per-head block-2 scores/weights.

    Keys follow the manuscript notation:
      x1      = LN1(emb)              (Layer-1 attention input, x^{(1)})
      attn1_w = Layer-1 attention weights [B,H,T,T]
      h1bar   = LN2(post_attn_1)      (\bar h^{(1)})
      h1      = block1 output         (h^{(1)})
      x2      = LN3(h1)               (Layer-2 attention input, x^{(2)})
      scores2 = Layer-2 per-head scores [B,H,T,T] (float32)
      attn2_w = Layer-2 attention weights [B,H,T,T]
      o2      = Layer-2 attention update (full, all heads + bias)
      post2   = h1 + o2
      ln4     = LN4(post2)
      m2      = MLP2 output
      h2      = final residual state
      pred    = position prediction (normalized units)
    """
    model.eval()
    out, _ = model(idx, targets=None, capture_taps=True)
    taps = model.get_all_taps()
    attn1_w, attn2_w = model.get_attention_weights()
    hw2 = head_weights(model.block2.attn)
    scores2, _ = attn_scores_weights(taps["block2_ln1"], hw2)
    return {
        "emb": taps["embeddings"],
        "x1": taps["block1_ln1"],
        "attn1_w": attn1_w,
        "post1": taps["block1_post_attn"],
        "h1bar": taps["block1_ln2"],
        "h1": taps["block1_out"],
        "x2": taps["block2_ln1"],
        "scores2": scores2,
        "attn2_w": attn2_w,
        "o2": taps["block2_attn"],
        "post2": taps["block2_post_attn"],
        "ln4": taps["block2_ln2"],
        "m2": taps["block2_mlp"],
        "h2": taps["block2_out"],
        "pred": out.squeeze(-1),
    }


def denorm_pred(pred: torch.Tensor, context_length: int) -> torch.Tensor:
    """Model predicts position / (L_train - 1); convert to absolute position
    using the evaluated context length (matches training normalization when
    evaluated at the training length)."""
    return pred * (context_length - 1)


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def run_dir(run_id: str) -> Path:
    d = RESULTS_ROOT / run_id
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def save_config(d: Path, meta: dict, extra: dict):
    cfg = dict(meta)
    cfg.update(extra)
    cfg["git_commit"] = git_commit()
    with open(d / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def update_summary(d: Path, section: str, payload: dict):
    """Merge a section into summary.json (stable keys for manuscript numbers)."""
    path = d / "summary.json"
    summary = {}
    if path.exists():
        summary = json.loads(path.read_text())
    summary[section] = payload
    path.write_text(json.dumps(summary, indent=2, default=str))


def bootstrap_ci(per_seq_values: np.ndarray, n_boot: int = 1000, seed: int = 0,
                 stat=np.mean) -> Tuple[float, float, float]:
    """Bootstrap whole sequences (axis 0). Returns (stat, lo95, hi95)."""
    vals = per_seq_values[np.isfinite(per_seq_values)] if per_seq_values.ndim == 1 \
        else per_seq_values
    rng = np.random.default_rng(seed)
    n = len(vals)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = stat(vals[idx])
    return float(stat(vals)), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


POSITION_BINS = {"early": (1, 15), "middle": (16, 127), "late": (128, 1023)}


def position_bins(L: int) -> Dict[str, Tuple[int, int]]:
    if L >= 1024:
        return POSITION_BINS
    return {"early": (1, 15), "middle": (16, min(127, L - 1)),
            "late": (min(128, L - 1), L - 1)}
