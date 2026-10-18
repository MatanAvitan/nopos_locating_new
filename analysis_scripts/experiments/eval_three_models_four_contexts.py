"""Evaluate FULL-12H, ATTN2-1H, ATTN2-12H at L=128/1024/8192/32768.

For each checkpoint, computes:
  1) position_regression R² (linear pos_head on full forward pass)
  2) post_attn2 Linear R² (Ridge probe on raw Block-2 attention output)
  3) post_attn2 MLP R²    (2-hidden MLP probe on raw Block-2 attention output)

Writes results as JSON and prints a summary table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

ROOT = Path("/home/nlp/matan_avitan/git/nopos_locating_new")
sys.path.insert(0, str(ROOT / "nanoGPT"))
from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOS = 50256
NSEQ = 500
SEED = 42

# Checkpoint table: (label, family, L, path)
CHECKPOINTS = [
    # FULL-12H
    ("FULL-12H_128",   "FULL-12H",  128,   ROOT / "nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt"),
    ("FULL-12H_1024",  "FULL-12H",  1024,  ROOT / "nanoGPT/out-mechanism-R0-1024/R0/nuacla0w/best_ckpt.pt"),
    ("FULL-12H_8192",  "FULL-12H",  8192,  ROOT / "nanoGPT/out-mechanism-R0-8192/R0/lwrgnltx/best_ckpt.pt"),
    ("FULL-12H_32768", "FULL-12H",  32768, ROOT / "nanoGPT/out-mechanism-R0-32k/R0/5pey0fzk/best_ckpt.pt"),
    # ATTN2-1H
    ("ATTN2-1H_128",   "ATTN2-1H",  128,   ROOT / "nanoGPT/out-2layer-mechanism-r2-1head-attnonly-fullblock-40k/R2/o4w7v8dv/best_ckpt.pt"),
    ("ATTN2-1H_1024",  "ATTN2-1H",  1024,  ROOT / "nanoGPT/out-mechanism-R2-1024/R2/t72g9e8p/best_ckpt.pt"),
    ("ATTN2-1H_8192",  "ATTN2-1H",  8192,  ROOT / "nanoGPT/out-mechanism-R2-8192/R2/czld6m70/best_ckpt.pt"),
    ("ATTN2-1H_32768", "ATTN2-1H",  32768, ROOT / "nanoGPT/out-mechanism-R2-32k/R2/5r01gjjo/best_ckpt.pt"),
    # ATTN2-12H
    ("ATTN2-12H_128",   "ATTN2-12H", 128,   ROOT / "nanoGPT/out-mechanism-R2-12h-128/R2/375mmb9k/best_ckpt.pt"),
    ("ATTN2-12H_1024",  "ATTN2-12H", 1024,  ROOT / "nanoGPT/out-mechanism-R2-12h-1024/R2/pb1sxvzm/best_ckpt.pt"),
    ("ATTN2-12H_8192",  "ATTN2-12H", 8192,  ROOT / "nanoGPT/out-mechanism-R2-12h-8192/R2/pblr79la/best_ckpt.pt"),
    ("ATTN2-12H_32768", "ATTN2-12H", 32768, ROOT / "nanoGPT/out-mechanism-R2-12h-32k/R2/8o6lqqwh/best_ckpt.pt"),
    # (ATTN2-12H_128 and ATTN2-12H_1024: pending training)
]


class MLPProbe(nn.Module):
    def __init__(self, d_in=768, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    config = TwoLayerMechanismConfig(
        block_size=cfg["block_size"], vocab_size=cfg["vocab_size"],
        n_embd=cfg["n_embd"], n_head=cfg["n_head"],
        dropout=0.0, norm_type=cfg["norm_type"], bias=True, use_regression=True,
        use_flash=True,
    )
    model = TwoLayerMechanismModel(config)
    model.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    )
    model.to(DEVICE).eval()
    return model, cfg


def r2_cod(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def evaluate_one(label, ckpt_path, seq_len):
    print(f"\n{'=' * 60}")
    print(f"  {label}  ({ckpt_path.name})")
    print(f"{'=' * 60}")

    model, cfg = load_model(ckpt_path)
    block_size = cfg["block_size"]
    assert block_size == seq_len, f"seq_len mismatch: {seq_len} vs {block_size}"
    scale = max(block_size - 1, 1)

    data = np.memmap(ROOT / "nanoGPT/data/openwebtext/train.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(data) - (block_size - 1), size=NSEQ)
    seqs = np.empty((NSEQ, block_size), dtype=np.int64)
    seqs[:, 0] = BOS
    for i, s in enumerate(starts):
        seqs[i, 1:] = data[s:s + block_size - 1].astype(np.int64)

    # Batch size and position subsampling
    if block_size <= 128:
        bs = 16
    elif block_size <= 1024:
        bs = 4
    elif block_size <= 8192:
        bs = 2
    else:  # 32k
        bs = 1
    stride = max(1, block_size // 2048)
    pos_idx = torch.arange(0, block_size, stride)

    y_true_all, y_pred_all = [], []
    positions_full = np.arange(block_size, dtype=np.float32)

    for batch_start in range(0, NSEQ, bs):
        batch_end = min(batch_start + bs, NSEQ)
        tokens = torch.from_numpy(seqs[batch_start:batch_end]).to(DEVICE)
        actual_bs = tokens.shape[0]
        with torch.no_grad():
            e = model.wte(tokens)
            x1 = model.block1(e, capture_taps=False)
            x2 = model.block2(x1, capture_taps=False)
            ln_f_out = model.ln_f(x2)
            preds = model.pos_head(ln_f_out).squeeze(-1) * scale

        y_pred_all.append(preds.cpu().numpy().reshape(-1))
        y_true_all.append(np.tile(positions_full, actual_bs))

    # Position regression (full forward pass, linear pos_head)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    r2_pos = r2_cod(y_true, y_pred)

    print(f"  position_regression R² = {r2_pos:.4f}")
    del model
    torch.cuda.empty_cache()
    return {
        "label": label,
        "seq_len": seq_len,
        "checkpoint": str(ckpt_path),
        "position_regression_r2": r2_pos,
    }


def main():
    results = {}
    for label, family, seq_len, ckpt_path in CHECKPOINTS:
        if not ckpt_path.exists():
            print(f"[SKIP] {label}: missing {ckpt_path}")
            continue
        try:
            results[label] = evaluate_one(label, ckpt_path, seq_len)
        except Exception as e:
            print(f"[ERROR] {label}: {e}")
            import traceback
            traceback.print_exc()

    out_dir = ROOT / "results/three_models_four_contexts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nSaved → {out_path}")

    # Print summary table
    print("\n\n" + "=" * 90)
    print(f"{'Model':16s} | {'L':>6s} | {'pos_reg R²':>11s}")
    print("=" * 50)
    for label, r in results.items():
        print(f"{label:16s} | {r['seq_len']:>6d} | {r['position_regression_r2']:>11.4f}")


if __name__ == "__main__":
    main()
