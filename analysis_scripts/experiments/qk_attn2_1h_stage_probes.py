"""
Per-position, per-stage decodability of position in ATTN2-1H.

IMPORTANT framing: in this checkpoint (r2_attn_head_only=True) ONLY block2.attn
and pos_head are trained; the Layer-2 MLP, both block-2 LayerNorms, and ln_f are
FROZEN at random init. So the MLP does not *learn* an inverse -- it is a fixed
nonlinear feature map, and the trained attention places the positional trajectory
where that fixed nonlinearity (+ trained linear head) linearizes it.

We isolate every intermediate state between the attention output and the head:

    o   = o_i^(2)              block2_attn        (attention output)
    htd = h~_i^(2)             block2_post_attn   (= h^(1) + o^(2), residual add)
    ln  = LN(h~_i^(2))         block2_ln2         (pre-MLP LayerNorm)
    m   = m_i^(2)              block2_mlp         (frozen-MLP output)
    h   = h_i^(2)              block2_out         (= h~ + m)

For each state we fit a fresh linear ridge probe (train/test split by sample) and
report per-position test MAE.  We add:
  - a NONLINEAR probe (small MLP) on the raw attention output o, to show position
    is already present in o but only nonlinearly accessible;
  - a zero-shot no-MLP run using the ORIGINAL trained head on h~ (set_post_attn_head),
    to test whether the original circuit *requires* the MLP.

Figure: Panel A absolute MAE(i); Panel B relative MAE(i)/i.
Output: results/qk_attn2_1h/stage_probes.{png,pdf} + metrics json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "analysis_scripts" / "experiments"))
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))

import qk_attn2_1h_analysis as Q

STAGES = [
    ("o_attn", "block2_attn", "$o^{(2)}$ attn out"),
    ("htilde", "block2_post_attn", r"$\tilde h^{(2)}$ resid"),
    ("ln_pre_mlp", "block2_ln2", r"LN$(\tilde h^{(2)})$"),
    ("mlp_out", "block2_mlp", "$m^{(2)}$ MLP out"),
    ("h_block_out", "block2_out", "$h^{(2)}$ block out"),
]


@torch.no_grad()
def collect(model, data, T, bs, nb):
    store = {k: [] for k, _, _ in STAGES}
    preds, post_attn_preds = [], []
    Tblk = model.config.block_size
    torch.manual_seed(0)
    seeds = torch.randint(0, 2**31 - 1, (nb,))
    for s in seeds:
        torch.manual_seed(int(s))
        tok = Q.sample_tokens(data, T, bs)
        out, _ = model(tok, capture_taps=True)
        taps = model.get_all_taps()
        for k, tap, _ in STAGES:
            store[k].append(taps[tap].half().cpu().numpy())
        preds.append(out.squeeze(-1).float().cpu().numpy() * Tblk)
    # zero-shot: original head on post-attn residual (no MLP)
    model.set_post_attn_head(True)
    for s in seeds:
        torch.manual_seed(int(s))
        tok = Q.sample_tokens(data, T, bs)
        out, _ = model(tok, capture_taps=False)
        post_attn_preds.append(out.squeeze(-1).float().cpu().numpy() * Tblk)
    model.set_post_attn_head(False)
    store = {k: np.concatenate(v, 0) for k, v in store.items()}
    return store, np.concatenate(preds, 0), np.concatenate(post_attn_preds, 0)


def ridge_per_pos(Xtr, ytr, Xte, yte_pos, T, lam):
    """Fit ridge on standardized features; return per-position test MAE (pos 1..T-1)."""
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1), Xtr.dtype)], 1)
    Xte = np.concatenate([Xte, np.ones((len(Xte), 1), Xte.dtype)], 1)
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=Xtr.dtype)
    w = np.linalg.solve(A, Xtr.T @ ytr)
    yp = Xte @ w
    n_te = len(yp) // (T - 1)
    err = np.abs(yp.reshape(n_te, T - 1) - np.arange(1, T))
    return err.mean(0), yp  # per-position MAE, flat preds


def nonlinear_probe_on_o(o, pos_norm, ntr, T, device, hidden=256, steps=2000):
    """Train a small MLP probe (o -> normalized position) and return per-pos MAE."""
    N = o.shape[0]
    Xtr = torch.tensor(o[:ntr, 1:].reshape(-1, o.shape[2]), dtype=torch.float32)
    Xte = torch.tensor(o[ntr:, 1:].reshape(-1, o.shape[2]), dtype=torch.float32)
    ytr = torch.tensor(np.tile(pos_norm, ntr), dtype=torch.float32)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr = ((Xtr - mu) / sd).to(device); Xte = ((Xte - mu) / sd).to(device)
    ytr = ytr.to(device)
    net = nn.Sequential(nn.Linear(o.shape[2], hidden), nn.GELU(),
                        nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    bs = 8192
    for _ in range(steps):
        idx = torch.randint(0, Xtr.shape[0], (bs,), device=device)
        opt.zero_grad()
        loss = ((net(Xtr[idx]).squeeze(-1) - ytr[idx]) ** 2).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        yp = []
        for i in range(0, Xte.shape[0], 65536):
            yp.append(net(Xte[i:i + 65536]).squeeze(-1).cpu())
        yp = torch.cat(yp).numpy() * T  # back to positions
    n_te = (Xte.shape[0]) // (T - 1)
    err = np.abs(yp.reshape(n_te, T - 1) - np.arange(1, T))
    return err.mean(0)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=Q.DEFAULT_CKPT)
    p.add_argument("--data_dir", default=Q.DEFAULT_DATA)
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_batches", type=int, default=12)
    p.add_argument("--save_dir", default="results/qk_attn2_1h")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Q.load_model(ROOT_DIR / args.ckpt)
    data = Q.load_data(ROOT_DIR / args.data_dir)
    T = min(args.T, model.config.block_size)

    store, preds, post_attn_preds = collect(model, data, T, args.batch_size, args.n_batches)
    N = preds.shape[0]; ntr = N // 2
    pos = np.arange(1, T).astype(float)
    pos_norm = (np.arange(T) / T)[1:]  # normalized targets, pos 1..T-1

    per_pos = {}
    flat_pos_tr = np.tile(pos, ntr)
    for k, _, _ in STAGES:
        X = store[k].astype(np.float32)
        Xtr = X[:ntr, 1:].reshape(-1, X.shape[2])
        Xte = X[ntr:, 1:].reshape(-1, X.shape[2])
        lam = 10.0
        mae_pos, _ = ridge_per_pos(Xtr, flat_pos_tr, Xte, None, T, lam)
        per_pos[k] = mae_pos
        del X, Xtr, Xte

    # nonlinear probe on o
    per_pos["o_attn_nonlinear"] = nonlinear_probe_on_o(
        store["o_attn"].astype(np.float32), pos_norm, ntr, T, device
    )

    # model head (post-MLP, trained) per-position, test split
    err_head = np.abs(preds[ntr:][:, 1:] - pos)
    per_pos["model_head"] = err_head.mean(0)
    # zero-shot no-MLP (original head on post-attn residual)
    err_noMLP = np.abs(post_attn_preds[ntr:][:, 1:] - pos)
    per_pos["zeroshot_noMLP_head"] = err_noMLP.mean(0)

    # ---------------- figure ----------------
    save_dir = ROOT_DIR / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    def smooth(y, w=9):
        k = np.ones(w) / w
        return np.convolve(y, k, mode="same")

    order = [
        ("o_attn", "C0", "linear on $o^{(2)}$"),
        ("o_attn_nonlinear", "C9", "nonlinear on $o^{(2)}$"),
        ("htilde", "C1", r"linear on $\tilde h^{(2)}$"),
        ("ln_pre_mlp", "C2", r"linear on LN$(\tilde h^{(2)})$"),
        ("mlp_out", "C4", "linear on $m^{(2)}$"),
        ("h_block_out", "C3", "linear on $h^{(2)}$"),
        ("model_head", "k", "model head (post-MLP)"),
    ]
    for k, c, lab in order:
        ax[0].plot(pos, smooth(per_pos[k]), color=c, lw=1.4, label=lab)
        ax[1].plot(pos, smooth(per_pos[k] / pos), color=c, lw=1.4, label=lab)
    ax[0].axvline(768, color="grey", ls=":", lw=1)
    ax[0].set_title("Panel A: absolute MAE by position"); ax[0].set_xlabel("position $i$")
    ax[0].set_ylabel("per-position MAE"); ax[0].set_ylim(0, 120); ax[0].legend(fontsize=8)
    ax[1].axvline(768, color="grey", ls=":", lw=1)
    ax[1].set_title("Panel B: relative MAE / $i$ by position"); ax[1].set_xlabel("position $i$")
    ax[1].set_ylabel("MAE$(i)/i$"); ax[1].set_ylim(0, 1.0); ax[1].legend(fontsize=8)
    fig.suptitle("ATTN2-1H staged decoding (Layer-2 MLP is FROZEN random): where position becomes linearly readable", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = save_dir / "stage_probes.png"
    fig.savefig(png, dpi=150, bbox_inches="tight"); fig.savefig(save_dir / "stage_probes.pdf", bbox_inches="tight")
    plt.close(fig)

    def windows(y):
        return {f"{a}-{b}": float(np.mean(y[a - 1:b - 1])) for a, b in [(1, 64), (64, 256), (256, 768), (768, 1024)]}
    summary = {k: {"overall_mae": float(np.mean(v)), "windows_mae": windows(v)} for k, v in per_pos.items()}
    with open(save_dir / "stage_probes_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", png)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
