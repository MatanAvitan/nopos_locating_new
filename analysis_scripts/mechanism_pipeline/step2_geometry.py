"""Step 2 (exps.md): OV endpoint separation and low-dimensional geometry.

Consumes ref_stats.pt + saved eval arrays + model weights. No GPU needed.

Usage: python step2_geometry.py --run-id attn2_1h_L1024 --model attn2_1h
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_ROOT, head_weights, load_model, update_summary


def principal_angles(U: np.ndarray, V: np.ndarray):
    """Principal angles (deg) between column spans of U and V."""
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    s = np.linalg.svd(Qu.T @ Qv, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1))).tolist()


def cos(a, b, eps=1e-30):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    out = RESULTS_ROOT / args.run_id
    ref = torch.load(out / "ref_stats.pt", map_location="cpu", weights_only=False)
    model, meta = load_model(args.model, device="cpu")
    hw2 = head_weights(model.block2.attn)
    H = model.config.n_head
    d = model.config.n_embd

    w_b = ref["w_bos_2"].numpy()          # [H,d]
    w_nb = ref["w_nonbos_2"].numpy()      # [H,d]
    w_nb_h1 = ref["w_nonbos_2_half1"].numpy()
    w_nb_h2 = ref["w_nonbos_2_half2"].numpy()
    mu_j = ref["mu_j"].numpy()            # [H,L,d]
    dw = w_nb - w_b

    per_head = {}
    for h in range(H):
        B_ov = (hw2.W_o[h].double() @ hw2.W_v[h].double()).numpy()  # [d,d]
        U, S, Vt = np.linalg.svd(B_ov)
        span_mech = np.stack([w_b[h], w_nb[h]], 1)  # [d,2]

        # 2.2 position-specific mean OV images vs global vector
        L = mu_j.shape[1]
        dist_j = np.linalg.norm(mu_j[h, 1:] - w_nb[h], axis=-1) / np.linalg.norm(w_nb[h])
        cos_j = (mu_j[h, 1:] @ w_nb[h]) / (
            np.linalg.norm(mu_j[h, 1:], axis=-1) * np.linalg.norm(w_nb[h]) + 1e-30)

        per_head[f"head{h}"] = {
            "norm_w_bos": float(np.linalg.norm(w_b[h])),
            "norm_w_nonbos": float(np.linalg.norm(w_nb[h])),
            "norm_dw": float(np.linalg.norm(dw[h])),
            "cos_bos_nonbos": cos(w_b[h], w_nb[h]),
            "angle_bos_nonbos_deg": float(np.degrees(np.arccos(
                np.clip(cos(w_b[h], w_nb[h]), -1, 1)))),
            "normalized_separation": float(np.linalg.norm(dw[h]) /
                (np.linalg.norm(w_b[h]) + np.linalg.norm(w_nb[h]))),
            "w_nonbos_half_cos": cos(w_nb_h1[h], w_nb_h2[h]),
            "w_nonbos_half_relerr": float(np.linalg.norm(w_nb_h1[h] - w_nb_h2[h])
                                          / np.linalg.norm(w_nb[h])),
            "ov_singular_values_top8": S[:8].tolist(),
            "align_wbos_u1": abs(cos(w_b[h], U[:, 0])),
            "align_wnonbos_u1": abs(cos(w_nb[h], U[:, 0])),
            "align_dw_u1": abs(cos(dw[h], U[:, 0])),
            "align_wbos_u2": abs(cos(w_b[h], U[:, 1])),
            "align_wnonbos_u2": abs(cos(w_nb[h], U[:, 1])),
            "align_dw_u2": abs(cos(dw[h], U[:, 1])),
            "principal_angles_mech_vs_top2svd_deg":
                principal_angles(span_mech, U[:, :2]),
            "mu_j_dist_rel_quantiles": {
                q: float(np.quantile(dist_j, float(q))) for q in
                ["0.05", "0.25", "0.5", "0.75", "0.95"]},
            "mu_j_cos_quantiles": {
                q: float(np.quantile(cos_j, float(q))) for q in
                ["0.05", "0.25", "0.5", "0.75", "0.95"]},
        }

    # trajectory dimensionality (position-conditioned mean, bias-centered)
    traj = np.load(out / "o2c_mean_traj.npy")     # [T,d]
    ohead = np.load(out / "ohead_mean_traj.npy")  # [H,T,d]

    def traj_dim(X):
        Xc = X - X.mean(0)
        s = np.linalg.svd(Xc, compute_uv=False)
        ev = s ** 2 / (s ** 2).sum()
        return {"var_frac_1d": float(ev[0]), "var_frac_2d": float(ev[:2].sum()),
                "var_frac_3d": float(ev[:3].sum())}

    summed = {
        "summed_trajectory": traj_dim(traj[1:]),
        "per_head_trajectory": {f"head{h}": traj_dim(ohead[h, 1:]) for h in range(H)},
    }
    # cross-head geometry of Delta w (Step 4.8)
    if H > 1:
        dwn = dw / np.linalg.norm(dw, axis=-1, keepdims=True)
        summed["dw_pairwise_cos"] = (dwn @ dwn.T).tolist()
        s = np.linalg.svd(dw - dw.mean(0), compute_uv=False)
        ev = s ** 2 / (s ** 2).sum()
        summed["dw_pca_var_frac"] = ev[:6].tolist()

    update_summary(out, "step2_geometry", {"per_head": per_head, **summed})
    print(json.dumps({"per_head_head0": per_head["head0"], **{k: v for k, v in summed.items() if k == "summed_trajectory"}}, indent=2))


if __name__ == "__main__":
    main()
