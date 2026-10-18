"""Diagnostic: in-window <|endoftext|> tokens (id 50256 at j >= 1) as content sinks.

For one model and one layer, on validation: how often windows contain them,
their beta (primary convention, as in cc_screen) against ordinary keys, the
attention each head of the layer gives them from late queries, and the
position-1 sink's beta distribution.

Usage: python cc_eot.py --model 1h_s42 --layer 1 --device cuda:0
"""

import argparse

import numpy as np
import torch

from cc_common import BOS_TOKEN_ID, OUT_ROOT, SPECS, Capture, jdump, load_net, splits_for


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS))
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    a = ap.parse_args()
    net, _ = load_net(a.model, a.device)
    sp, _ = splits_for(a.model)
    L, l, H = net.spec.L, a.layer, net.H
    b_ln = net.ln_params(net.blocks[l].ln_1)[1]
    x0s, n = 0, 0
    for s in range(0, 160, a.batch_size):
        idx = net.batch(sp["calibration"][s:s + a.batch_size]); cap = Capture(want={("x_attn", l)}); net.forward(idx, cap, stop_after=l)
        x0s = x0s + cap.store[("x_attn", l)][:, 0].double().sum(0); n += idx.shape[0]
    d = x0s / n - b_ln
    be, bo, b1, att_e, att_o, cnt, att_e_any = [], [], [], [], [], [], []
    for s in range(0, len(sp["validation"]), a.batch_size):
        idx = net.batch(sp["validation"][s:s + a.batch_size]); cap = Capture(want={("x_attn", l), ("weights", l)}); net.forward(idx, cap, stop_after=l)
        x = cap.store[("x_attn", l)].double(); A = cap.store[("weights", l)].double()
        beta = ((x - b_ln) @ d) / (d @ d)
        eot = (idx == BOS_TOKEN_ID); eot[:, 0] = False
        ordm = ~eot; ordm[:, :2] = False
        be.append(beta[eot].cpu()); bo.append(beta[ordm].cpu()); b1.append(beta[:, 1].cpu())
        cnt.append(eot.sum(1).cpu())
        q = torch.arange(L, device=idx.device)
        late = q >= min(128, L // 2)
        # per-key attention from late queries that can see the key, by head
        vis = (q[None, :] <= q[:, None])                                      # [T(query), T(key)]
        for bi in range(idx.shape[0]):
            e = torch.nonzero(eot[bi]).flatten()
            if len(e) == 0:
                continue
            Al = A[bi][:, late]                                               # [H, nq, T]
            vl = vis[late]
            for j in e.tolist():
                m_ = vl[:, j]
                if m_.sum() == 0:
                    continue
                att_e.append(Al[:, m_, j].mean(1).cpu())                     # [H]
                o = ordm[bi].clone(); o[j:] = False
                o_idx = torch.nonzero(o).flatten()
                if len(o_idx):
                    att_o.append(Al[:, m_][:, :, o_idx].mean((1, 2)).cpu())
    be, bo, b1 = torch.cat(be).numpy(), torch.cat(bo).numpy(), torch.cat(b1).numpy()
    cnt = torch.cat(cnt).numpy()
    q = lambda v: {p: float(np.quantile(v, p)) for p in (0.01, 0.1, 0.5, 0.9, 0.99)} if len(v) else None
    out = {"model": a.model, "layer": l, "n_val": len(cnt), "frac_windows_with_eot": float((cnt > 0).mean()),
           "mean_eot_per_window": float(cnt.mean()),
           "beta_eot": q(be), "beta_ordinary": q(bo), "beta_pos1": q(b1),
           "attn_per_key_from_late_queries": None if not att_e else {
               "eot_mean_by_head": torch.stack(att_e).mean(0).tolist(),
               "ordinary_mean_by_head": torch.stack(att_o).mean(0).tolist() if att_o else None}}
    jdump(out, OUT_ROOT / "eot" / f"{a.model}_L{l}.json")
    print(out)


if __name__ == "__main__":
    main()
