"""Cross-fitted probes and decoders (exps.md Steps 1.5, 5.2, 5.3, 5.4).

All probes/decoders/constants are fit on the CALIBRATION split and evaluated
on the EVALUATION split (same splits as extract.py, stored in sequence_ids.npz).

Usage: python probes.py --run-id attn2_1h_L1024 --model attn2_1h --context-length 1024
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS_ROOT, attn_bias_vector, forward_capture,
                    head_weights, iter_batches, load_model, position_bins,
                    update_summary)

STATE_KEYS = ["h1bar", "post1", "h1", "o2", "post2", "ln4", "m2", "h2"]
NONLINEAR_STATES = ["h1", "o2", "post2"]


def mlp_hidden(model, ln4):
    return model.block2.mlp.gelu(model.block2.mlp.c_fc(ln4))


def stylized_proxy(x1):
    T = x1.shape[1]
    i_idx = torch.arange(T, device=x1.device, dtype=x1.dtype)
    return x1.cumsum(1) / (i_idx[None, :, None] + 1).sqrt()


def get_states(model, cap):
    s = {k: cap[k] for k in ["h1bar", "post1", "h1", "o2", "post2", "ln4", "m2", "h2"]}
    s["proxy"] = stylized_proxy(cap["x1"])
    s["mlp_hidden"] = mlp_hidden(model, cap["ln4"])
    return s


class Ridge:
    def __init__(self, dim, device, lam=1e-4):
        self.A = torch.zeros(dim + 1, dim + 1, device=device, dtype=torch.float64)
        self.b = torch.zeros(dim + 1, device=device, dtype=torch.float64)
        self.lam = lam
        self.w = None

    def add(self, X, y):
        X1 = torch.cat([X.double().reshape(-1, X.shape[-1]),
                        torch.ones(X.numel() // X.shape[-1], 1,
                                   device=X.device, dtype=torch.float64)], 1)
        self.A += X1.T @ X1
        self.b += X1.T @ y.double().reshape(-1)

    def solve(self):
        n = self.A.shape[0]
        reg = self.lam * torch.diag(self.A).mean() * torch.eye(
            n, device=self.A.device, dtype=torch.float64)
        reg[-1, -1] = 0
        self.w = torch.linalg.solve(self.A + reg, self.b)
        return self.w

    def predict(self, X):
        X1 = torch.cat([X.double(), torch.ones(*X.shape[:-1], 1,
                        device=X.device, dtype=torch.float64)], -1)
        return X1 @ self.w


class TinyMLP(torch.nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_tiny_mlp(X, y, device, epochs=200, lr=1e-3, batch=65536, seed=0):
    """X: [N, d] float32 tensor (cpu), y: [N]."""
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    net = TinyMLP(X.shape[-1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    N = len(X)
    y_mu, y_sd = y.mean(), y.std().clamp_min(1e-6)
    for ep in range(epochs):
        perm = torch.randperm(N)[:min(N, 4 * batch)]
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            xb = ((X[idx] - mu) / sd).to(device)
            yb = ((y[idx] - y_mu) / y_sd).to(device)
            loss = F.mse_loss(net(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    def predict(Xe, bs=131072):
        outs = []
        with torch.no_grad():
            for s in range(0, len(Xe), bs):
                xb = ((Xe[s:s + bs].cpu() - mu) / sd).to(device)
                outs.append((net(xb).cpu() * y_sd + y_mu))
        return torch.cat(outs)
    return predict


def metrics(preds: np.ndarray, L: int, prefix: str):
    """preds: [n_seq, T] absolute-position predictions."""
    tgt = np.arange(L)[None, :].repeat(len(preds), 0).astype(np.float64)
    p, t = preds.reshape(-1), tgt.reshape(-1)
    finite = np.isfinite(p)
    p, t = p[finite], t[finite]
    ss_res = ((p - t) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    rho_seq = [spearmanr(preds[b][np.isfinite(preds[b])],
                         np.arange(L)[np.isfinite(preds[b])]).statistic
               for b in range(min(len(preds), 400))]
    out = {
        "r2": float(r2),
        "pearson_r2": float(np.corrcoef(p, t)[0, 1] ** 2),
        "mae": float(np.abs(p - t).mean()),
        "median_seq_spearman": float(np.nanmedian(rho_seq)),
    }
    for name, (a, b) in position_bins(L).items():
        sel = (tgt >= a) & (tgt <= b) & np.isfinite(preds)
        out[f"mae_{name}"] = float(np.abs(preds[sel] - tgt[sel]).mean())
        out[f"rel_err_{name}"] = float(
            (np.abs(preds[sel] - tgt[sel]) / np.maximum(tgt[sel], 1)).mean())
    return {prefix: out}


@torch.no_grad()
def collect(model, offsets, L, batch_size, device, want_states, want_scalars,
            ref, cache_states=None):
    """One pass. Returns dict of concatenated scalar features and optionally
    caches fp16 states; calls callback-style ridge accumulation externally."""
    raise NotImplementedError  # replaced by inline logic in main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--context-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = RESULTS_ROOT / args.run_id
    L = args.context_length
    device = args.device

    model, meta = load_model(args.model, device)
    ref = torch.load(out / "ref_stats.pt", map_location="cpu", weights_only=False)
    splits = np.load(out / "sequence_ids.npz")
    H = model.config.n_head
    d = model.config.n_embd
    hw2 = head_weights(model.block2.attn)
    w_nb = ref["w_nonbos_2"].to(device)
    w_b = ref["w_bos_2"].to(device)
    dw = (w_nb - w_b).float()
    b_attn2 = ref["b_attn_2"].to(device).float()
    i_idx = torch.arange(L, dtype=torch.float64)

    lnf, head = model.ln_f, model.pos_head
    gamma = lnf.weight.float()
    beta = lnf.bias.float() if lnf.bias is not None else torch.zeros_like(gamma)
    w_read = head.weight.float().squeeze(0)
    b_read = float(head.bias) if head.bias is not None else 0.0
    wg = w_read * gamma

    probe_states = ["proxy", "h1bar", "post1", "h1", "o2", "post2", "ln4",
                    "mlp_hidden", "m2", "h2"]
    dims = {k: (4 * d if k == "mlp_hidden" else d) for k in probe_states}
    ridges = {k: Ridge(dims[k], device) for k in probe_states}
    # scalar ridges for 5.2/5.4
    scalar_feats = ["alpha", "Y", "term_h1", "term_o2", "terms_joint",
                    "alpha_analytic_input"]

    def scalar_features(cap):
        alpha = cap["attn2_w"][:, :, :, 0].float()             # [B,H,T]
        o2c = cap["o2"] - b_attn2[None, None, :]
        Y = torch.einsum("btd,hd->bht", o2c, dw)               # [B,H,T]
        h2 = cap["h2"]
        sd_h = (h2.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt().squeeze(-1)
        term_h1 = (cap["h1"] @ wg) / sd_h
        term_o2 = (cap["o2"] @ wg) / sd_h
        term_m2 = (cap["m2"] @ wg) / sd_h
        return alpha, Y, term_h1, term_o2, term_m2

    # ---------------- calibration pass ---------------------------------
    cal = {"alpha": [], "Y": [], "t1": [], "t2": [], "tm": []}
    cal_states = {k: [] for k in NONLINEAR_STATES}
    tgt_row = torch.arange(L, dtype=torch.float32)
    print("calibration pass...")
    for s, idx in iter_batches(splits["calibration"], L, args.batch_size, device):
        cap = forward_capture(model, idx)
        states = get_states(model, cap)
        y = tgt_row.to(device).expand(idx.shape[0], L)
        for k in probe_states:
            ridges[k].add(states[k], y)
        alpha, Y, t1, t2, tm = scalar_features(cap)
        cal["alpha"].append(alpha.cpu()); cal["Y"].append(Y.cpu())
        cal["t1"].append(t1.cpu()); cal["t2"].append(t2.cpu()); cal["tm"].append(tm.cpu())
        for k in NONLINEAR_STATES:
            cal_states[k].append(states[k].half().cpu())
        del cap, states
    cal = {k: torch.cat(v, 0) for k, v in cal.items()}
    cal_states = {k: torch.cat(v, 0) for k, v in cal_states.items()}
    for k in probe_states:
        ridges[k].solve()

    # scalar decoders fit on calibration ------------------------------
    n_cal = cal["alpha"].shape[0]
    tgt_cal = tgt_row[None, :].expand(n_cal, L)
    mask_cal = torch.ones(n_cal, L, dtype=torch.bool); mask_cal[:, 0] = False

    def flat(x):  # [n,H,T] -> [N,H] over masked tokens
        return x.permute(0, 2, 1)[mask_cal]

    Xa_cal = flat(cal["alpha"])
    XY_cal = flat(cal["Y"])
    y_cal = tgt_cal[mask_cal]
    # linear on alpha / Y (closed form via lstsq)
    def linfit(X, y):
        X1 = torch.cat([X.double(), torch.ones(len(X), 1, dtype=torch.float64)], 1)
        w = torch.linalg.lstsq(X1, y.double().unsqueeze(1)).solution.squeeze(1)
        return w
    w_lin_alpha = linfit(Xa_cal, y_cal).to(device)
    w_lin_Y = linfit(XY_cal, y_cal).to(device)
    # analytic: i = (1-a)/(a*c); c estimated per head as median cbar on cal
    a_cal = cal["alpha"].double().clamp(1e-9, 1 - 1e-9)     # [n,H,T]
    c_hat_bi = (1 - a_cal) / (a_cal * i_idx[None, None, :].clamp(min=1))
    c_star = c_hat_bi[:, :, 1:].median(0).values.median(-1).values  # [H]
    analytic_cal = (1 - a_cal) / (a_cal * c_star[None, :, None])
    best_head = int((analytic_cal - tgt_cal[:, None, :]).abs()
                    .nanmean((0, 2)).argmin())
    # tiny MLPs
    print("training tiny-MLP decoders...")
    mlp_alpha = train_tiny_mlp(Xa_cal.float(), y_cal, device)
    mlp_Y = train_tiny_mlp(XY_cal.float(), y_cal, device)
    # 5.4 term probes
    T1_cal, T2_cal, TM_cal = (cal["t1"][mask_cal], cal["t2"][mask_cal],
                              cal["tm"][mask_cal])
    w_t1 = linfit(T1_cal.unsqueeze(1), y_cal).to(device)
    w_t2 = linfit(T2_cal.unsqueeze(1), y_cal).to(device)
    w_t12 = linfit(torch.stack([T1_cal, T2_cal], 1), y_cal).to(device)
    w_t12m = linfit(torch.stack([T1_cal, T2_cal, TM_cal], 1), y_cal).to(device)

    # nonlinear probes on selected states
    nl_probes = {}
    for k in NONLINEAR_STATES:
        X = cal_states[k].float().reshape(-1, d)
        yv = tgt_cal.reshape(-1)
        print(f"training nonlinear probe on {k}...")
        nl_probes[k] = train_tiny_mlp(X, yv, device, epochs=60)

    # ---------------- evaluation pass -----------------------------------
    print("evaluation pass...")
    n_eval = len(splits["evaluation"])
    preds = {}

    def stash(name, arr):
        preds.setdefault(name, []).append(arr.detach().cpu().float().numpy())

    for s, idx in iter_batches(splits["evaluation"], L, args.batch_size, device):
        cap = forward_capture(model, idx)
        states = get_states(model, cap)
        for k in probe_states:
            stash(f"probe_linear_{k}", ridges[k].predict(states[k]))
        for k in NONLINEAR_STATES:
            B = idx.shape[0]
            p = nl_probes[k](states[k].float().reshape(-1, d).cpu())
            stash(f"probe_mlp_{k}", p.reshape(B, L))
        alpha, Y, t1, t2, tm = scalar_features(cap)
        af = alpha.permute(0, 2, 1).double()                    # [B,T,H]
        Yf = Y.permute(0, 2, 1).double()
        stash("decode_linear_alpha", af @ w_lin_alpha[:-1] + w_lin_alpha[-1])
        stash("decode_linear_Y", Yf @ w_lin_Y[:-1] + w_lin_Y[-1])
        a_c = alpha.double().clamp(1e-9, 1 - 1e-9)
        stash("decode_analytic_alpha",
              ((1 - a_c) / (a_c * c_star.to(a_c.device)[None, :, None]))[:, best_head])
        B = idx.shape[0]
        stash("decode_mlp_alpha", mlp_alpha(af.reshape(-1, H).float()).reshape(B, L))
        stash("decode_mlp_Y", mlp_Y(Yf.reshape(-1, H).float()).reshape(B, L))
        stash("decode_term_h1", t1.double() * w_t1[0] + w_t1[1])
        stash("decode_term_o2", t2.double() * w_t2[0] + w_t2[1])
        stash("decode_terms_joint",
              t1.double() * w_t12[0] + t2.double() * w_t12[1] + w_t12[2])
        stash("decode_terms_all",
              t1.double() * w_t12m[0] + t2.double() * w_t12m[1]
              + tm.double() * w_t12m[2] + w_t12m[3])
        stash("model_head", cap["pred"] * (meta["block_size_train"] - 1))
        del cap, states

    results = {}
    for name, chunks in preds.items():
        arr = np.concatenate(chunks, 0)
        np.save(out / f"probe_preds__{name}.npy", arr)
        results.update(metrics(arr, L, name))
    results["analytic_best_head"] = best_head
    results["c_star_per_head"] = c_star.tolist()
    update_summary(out, "probes_decoders", results)
    print(json.dumps({k: v for k, v in results.items()
                      if isinstance(v, dict) and "r2" in v}, indent=2))


if __name__ == "__main__":
    main()
