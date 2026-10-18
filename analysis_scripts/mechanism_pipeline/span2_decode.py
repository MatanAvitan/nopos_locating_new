"""P3 (exps.md): decode position from the two span{w_B, w_NB} coordinates.

Per layer-2 head, build an orthonormal basis of span{w_B, w_NB}
(u1 = Δw/||Δw||, u2 = Gram-Schmidt of w_B against u1) and decode position
from the 2H scalars <o²−b, u1>, <o²−b, u2> with a linear and a tiny-MLP
decoder, cross-fitted on the same calibration/evaluation splits as all
other probes. Comparators (already in mae_tables): probe_{linear,mlp}_o2
(full-state ceiling) and decode_{linear,mlp}_Y (1-coordinate Δw baseline).

Usage: python span2_decode.py --run-id attn2_full_L1024 --model attn2_full --context-length 1024
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_ROOT, forward_capture, iter_batches, load_model, update_summary
from probes import metrics, train_tiny_mlp


def linfit(X, y):
    X1 = torch.cat([X.double(), torch.ones(len(X), 1, dtype=torch.float64)], 1)
    return torch.linalg.lstsq(X1, y.double().unsqueeze(1)).solution.squeeze(1)


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
    w_b = ref["w_bos_2"].to(device).float()        # [H, d]
    w_nb = ref["w_nonbos_2"].to(device).float()    # [H, d]
    b_attn2 = ref["b_attn_2"].to(device).float()   # [d]

    dw = w_b - w_nb
    u1 = dw / dw.norm(dim=-1, keepdim=True)
    r = w_b - (w_b * u1).sum(-1, keepdim=True) * u1
    u2 = r / r.norm(dim=-1, keepdim=True)
    U = torch.stack([u1, u2], 1)                   # [H, 2, d]
    H = U.shape[0]

    def span_feats(cap):
        o2c = cap["o2"] - b_attn2[None, None, :]
        return torch.einsum("btd,hkd->bthk", o2c, U).reshape(
            *o2c.shape[:2], 2 * H)                 # [B, T, 2H]

    tgt_row = torch.arange(L, dtype=torch.float32)
    Xs, ys = [], []
    print("calibration pass...")
    with torch.no_grad():
        for s, idx in iter_batches(splits["calibration"], L, args.batch_size, device):
            cap = forward_capture(model, idx)
            f = span_feats(cap)[:, 1:, :]          # exclude position 0
            Xs.append(f.reshape(-1, 2 * H).cpu())
            ys.append(tgt_row[1:].repeat(idx.shape[0]))
            del cap
    X_cal, y_cal = torch.cat(Xs, 0), torch.cat(ys, 0)
    w_lin = linfit(X_cal, y_cal).to(device)
    print("training tiny-MLP on 2H span coordinates...")
    mlp = train_tiny_mlp(X_cal.float(), y_cal, device)

    print("evaluation pass...")
    preds = {"decode_span2_linear": [], "decode_span2_mlp": []}
    with torch.no_grad():
        for s, idx in iter_batches(splits["evaluation"], L, args.batch_size, device):
            cap = forward_capture(model, idx)
            f = span_feats(cap)
            B = idx.shape[0]
            preds["decode_span2_linear"].append(
                (f.double() @ w_lin[:-1] + w_lin[-1]).cpu().float().numpy())
            preds["decode_span2_mlp"].append(
                mlp(f.reshape(-1, 2 * H).float().cpu()).reshape(B, L).numpy())
            del cap

    results = {"span_cos_wb_u1": (w_b / w_b.norm(dim=-1, keepdim=True) * u1)
               .sum(-1).tolist()}
    for name, chunks in preds.items():
        arr = np.concatenate(chunks, 0)
        np.save(out / f"probe_preds__{name}.npy", arr)
        results.update(metrics(arr, L, name))
    update_summary(out, "span2_decode", results)
    print(json.dumps({k: v for k, v in results.items()
                      if isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
