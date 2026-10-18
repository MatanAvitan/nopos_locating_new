"""
What does the Layer-2 normalization geometry actually do to the positional code?

Decisive battery (ATTN2-1H, frozen LN/MLP, trained attn+head):

  (1) Probe h^(1) alone (block1_out) -- how much of the o->h~ residual gain
      (254 -> 126) is Layer-1 info already present.
  (2) Two-direction reconstruction:  o_hat = b_O + alpha*w_BOS + (1-alpha)*w_nonBOS,
      push h^(1)+o_hat through the REAL frozen LN and MLP, probe LN(.) and h_hat.
      If it matches the real LN(r)=40 / h=33, the BOS-nonBOS interpolation is
      sufficient and eta_i is inessential.
  (3) LayerNorm component decomposition (gain/bias are frozen at 1/0, so trivial):
        r,  C r (center only),  r/||r|| (normalize only),  Cr/||Cr|| (= full LN).
      Isolates centering vs sphere-normalization as the source of the jump.
  (4) LayerNorm Jacobian along the OV interpolation direction delta:
        |DLN_r[delta]| = sqrt(d)/||z|| * |(I - z z^T/||z||^2) C delta|,  z = C r.
      Plot vs position -- does LN amplify the shrinking d alpha/di in the tail?
  (5) Fresh linear probe on ln_f(h^(2)) -- isolate final norm from the trained head.

All probes: fresh ridge, train/test split by sample, per-position test MAE.
Output: results/qk_attn2_1h/layernorm_anatomy.{png,pdf} + metrics json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "analysis_scripts" / "experiments"))
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))

import qk_attn2_1h_analysis as Q


def ridge_per_pos(X, pos_full, ntr, T, lam=10.0):
    """X: [N,T,d] float32. Fresh ridge, split by sample, per-position test MAE."""
    Xtr = X[:ntr, 1:].reshape(-1, X.shape[2]).astype(np.float64)
    Xte = X[ntr:, 1:].reshape(-1, X.shape[2]).astype(np.float64)
    ytr = np.tile(np.arange(1, T, dtype=np.float64), ntr)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    Xte = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)
    yp = Xte @ w
    n_te = len(yp) // (T - 1)
    return np.abs(yp.reshape(n_te, T - 1) - np.arange(1, T)).mean(0)


def windows(y):
    return {f"{a}-{b}": float(np.mean(y[a - 1:b - 1])) for a, b in [(1, 64), (64, 256), (256, 768), (768, 1024)]}


@torch.no_grad()
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=Q.DEFAULT_CKPT)
    p.add_argument("--data_dir", default=Q.DEFAULT_DATA)
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_batches", type=int, default=10)
    p.add_argument("--save_dir", default="results/qk_attn2_1h")
    args = p.parse_args()

    model = Q.load_model(ROOT_DIR / args.ckpt)
    data = Q.load_data(ROOT_DIR / args.data_dir)
    T = min(args.T, model.config.block_size)
    d = model.config.n_embd

    W = model.block2.attn.c_attn.weight
    W_V = W[2 * d:].cpu()
    W_O = model.block2.attn.c_proj.weight.cpu()
    b_O = model.block2.attn.c_proj.bias.cpu()
    M_OV = (W_O @ W_V)  # write_j = ln1(x_j) @ M_OV^T

    h1L, oL, rL, ln1L, houtL, aL = [], [], [], [], [], []
    torch.manual_seed(0)
    for _ in range(args.n_batches):
        tok = Q.sample_tokens(data, T, args.batch_size)
        model(tok, capture_taps=True)
        t = model.get_all_taps()
        _, a2 = model.get_attention_weights()
        h1L.append(t["block1_out"].half().cpu())
        oL.append(t["block2_attn"].half().cpu())
        rL.append(t["block2_post_attn"].half().cpu())
        ln1L.append(t["block2_ln1"].half().cpu())
        houtL.append(t["block2_out"].half().cpu())
        aL.append(a2[:, 0, :, 0].float().cpu())
    h1 = torch.cat(h1L); o = torch.cat(oL); r = torch.cat(rL)
    ln1 = torch.cat(ln1L); hout = torch.cat(houtL); alpha = torch.cat(aL)
    model.cpu()  # apply frozen LN/MLP on CPU tensors
    N = h1.shape[0]; ntr = N // 2
    pos = np.arange(1, T)

    res = {}
    def probe(name, X):
        res[name] = ridge_per_pos(np.asarray(X, dtype=np.float32), None, ntr, T)
        print(f"  {name:22s} overall={res[name].mean():7.2f}  late={res[name][767:].mean():7.2f}")

    print("== (1) Layer-1 residual content & raw stages ==")
    probe("h1_layer1", h1.float())
    probe("o_attn", o.float())
    probe("r_htilde", r.float())

    print("== (3) LayerNorm decomposition ==")
    rf = r.float()
    Cr = rf - rf.mean(-1, keepdim=True)
    r_sphere = rf / (rf.norm(dim=-1, keepdim=True) + 1e-6)
    Cr_sphere = Cr / (Cr.norm(dim=-1, keepdim=True) + 1e-6) * (d ** 0.5)  # = full LN (gain=1,bias=0)
    ln_real = model.block2.ln_2(r.float())  # exact frozen LN
    probe("center_only_Cr", Cr)
    probe("normalize_only", r_sphere)
    probe("Cr_sphere_=LN", Cr_sphere)
    probe("LN_real_block2ln2", ln_real)

    print("== (2) two-direction reconstruction through frozen LN+MLP ==")
    writes = ln1.float() @ M_OV.t()              # [N,T,d]
    w_BOS = writes[:, 0:1, :]                     # [N,1,d]
    w_nB = writes[:, 1:, :].mean(1, keepdim=True) # [N,1,d]
    a = alpha.unsqueeze(-1)                       # [N,T,1]
    o_hat = b_O + a * w_BOS + (1 - a) * w_nB      # [N,T,d]
    # fidelity of the 1-D approximation
    cos = torch.nn.functional.cosine_similarity(o_hat.reshape(-1, d), o.float().reshape(-1, d)).mean().item()
    r_hat = h1.float() + o_hat
    ln_hat = model.block2.ln_2(r_hat)
    m_hat = model.block2.mlp(ln_hat)
    h_hat = r_hat + m_hat
    probe("recon_LN", ln_hat)
    probe("recon_h_postmlp", h_hat)
    res_cos = cos
    print(f"  [o_hat vs o cosine = {cos:.3f}]")

    print("== (5) final norm isolation ==")
    probe("h2_block_out", hout.float())
    probe("lnf_h2", model.ln_f(hout.float()))

    # ---- (4) LayerNorm Jacobian along delta ----
    delta = (w_BOS - w_nB).squeeze(1)            # [N,d] OV interpolation dir (per sample)
    Cdelta = delta - delta.mean(-1, keepdim=True)  # [N,d]
    z = Cr                                        # [N,T,d]
    znorm = z.norm(dim=-1, keepdim=True) + 1e-6
    # (I - zz^T/||z||^2) Cdelta  -> per (n,t): Cdelta_n - z*(z.Cdelta_n)/||z||^2
    zc = (z * Cdelta.unsqueeze(1)).sum(-1, keepdim=True)  # [N,T,1]
    proj = Cdelta.unsqueeze(1) - z * (zc / (znorm ** 2))
    jac = (d ** 0.5) / znorm.squeeze(-1) * proj.norm(dim=-1)  # [N,T]
    jac_pos = jac.mean(0).numpy()
    # effective post-LN positional signal per step = |DLN[delta]| * |d alpha/di|
    a_mean = alpha.mean(0).numpy()
    dalpha = np.abs(np.gradient(a_mean))
    eff_signal = jac_pos * dalpha

    # ---------------- figure ----------------
    save_dir = ROOT_DIR / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    def sm(y, w=9):
        return np.convolve(y, np.ones(w) / w, mode="same")
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    for k, c in [("o_attn", "C0"), ("r_htilde", "C1"), ("center_only_Cr", "C5"),
                 ("normalize_only", "C6"), ("LN_real_block2ln2", "C2")]:
        ax[0, 0].plot(pos, sm(res[k]), color=c, lw=1.4, label=k)
    ax[0, 0].set_title("(3) LayerNorm decomposition: per-position MAE")
    ax[0, 0].set_xlabel("position $i$"); ax[0, 0].set_ylabel("MAE"); ax[0, 0].set_ylim(0, 160)
    ax[0, 0].legend(fontsize=8)

    for k, c in [("o_attn", "C0"), ("h1_layer1", "C7"), ("r_htilde", "C1")]:
        ax[0, 1].plot(pos, sm(res[k]), color=c, lw=1.4, label=k)
    ax[0, 1].set_title("(1) Layer-1 residual content")
    ax[0, 1].set_xlabel("position $i$"); ax[0, 1].set_ylabel("MAE"); ax[0, 1].set_ylim(0, 300)
    ax[0, 1].legend(fontsize=8)

    for k, c, ls in [("LN_real_block2ln2", "C2", "-"), ("recon_LN", "C2", "--"),
                     ("h2_block_out", "C3", "-"), ("recon_h_postmlp", "C3", "--"),
                     ("lnf_h2", "k", "-")]:
        ax[1, 0].plot(pos, sm(res[k]), color=c, ls=ls, lw=1.4, label=k)
    ax[1, 0].set_title(f"(2,5) reconstruction (dashed) vs real (solid)\n$o$-recon cos={res_cos:.3f}")
    ax[1, 0].set_xlabel("position $i$"); ax[1, 0].set_ylabel("MAE"); ax[1, 0].set_ylim(0, 80)
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(pos, sm(jac_pos[1:]), color="C4", label=r"$|D\mathrm{LN}_r[\delta]|$")
    ax2 = ax[1, 1].twinx()
    ax2.plot(pos, sm(eff_signal[1:]), color="C8", lw=1.2, label="eff. signal $|DLN[\\delta]|\\,|d\\alpha/di|$")
    ax[1, 1].set_title("(4) LayerNorm sensitivity along $\\delta$ vs position")
    ax[1, 1].set_xlabel("position $i$"); ax[1, 1].set_ylabel(r"$|D\mathrm{LN}[\delta]|$", color="C4")
    ax2.set_ylabel("effective signal/step", color="C8")
    ax[1, 1].legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle("ATTN2-1H LayerNorm anatomy: centering vs normalization, two-direction sufficiency, Jacobian", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = save_dir / "layernorm_anatomy.png"
    fig.savefig(png, dpi=150, bbox_inches="tight"); fig.savefig(save_dir / "layernorm_anatomy.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {k: {"overall": float(v.mean()), "windows": windows(v)} for k, v in res.items()}
    summary["o_hat_cosine"] = float(res_cos)
    summary["jacobian_delta_at"] = {int(i): float(jac_pos[i]) for i in [16, 64, 128, 512, 1000] if i < len(jac_pos)}
    with open(save_dir / "layernorm_anatomy_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", png)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
