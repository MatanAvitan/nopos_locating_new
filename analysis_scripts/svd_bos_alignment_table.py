"""
SVD Alignment Analysis: BOS vs Others directions in write map B = W_O @ W_V.

Computes cosine alignment between top singular vectors of B and the
empirical BOS / others / contrast directions, plus singular value dominance ratios.

Key insight: with BOS at position 0, mean(B x_0) and mean(B x_j) are anti-correlated
(cos~-0.62). The position-encoding signal lives in both the shared and contrast directions.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).parent.parent / "nanoGPT"))
from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

BOS_TOKEN_ID = 50256


def load_model(checkpoint_path: str, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint.get("config", checkpoint.get("model_args", {}))
    valid_keys = set(TwoLayerMechanismConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = TwoLayerMechanismConfig(**filtered)
    model = TwoLayerMechanismModel(config)
    state_dict = checkpoint["model"]
    unwrapped = {}
    for k, v in state_dict.items():
        key = k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k
        unwrapped[key] = v
    model.load_state_dict(unwrapped)
    model.to(device)
    model.eval()
    return model, config


def load_owt_data(data_dir: str = "nanoGPT/data/openwebtext"):
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return val_data


def get_batch_with_bos(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    seqs = []
    for i in ix:
        after_bos = data[i : i + block_size - 1].astype(np.int64)
        seq = np.concatenate([[BOS_TOKEN_ID], after_bos])
        seqs.append(torch.from_numpy(seq))
    return torch.stack(seqs).to(device)


def get_write_map(model):
    attn = model.block2.attn
    c_attn_weight = attn.c_attn.weight
    d = c_attn_weight.shape[1]
    W_V = c_attn_weight[2 * d :, :]
    W_O = attn.c_proj.weight
    B = W_O @ W_V
    return B.detach()


def compute_bos_others_directions(
    model, data, B_torch, n_batches=50, batch_size=32, block_size=128, device="cuda"
):
    """Compute mean B*x_0 (BOS dir) and mean B*x_j for j>0 (others dir) from data."""
    d = B_torch.shape[0]
    bos_sum = torch.zeros(d, device=device)
    others_sum = torch.zeros(d, device=device)
    n_bos = 0
    n_others = 0

    with torch.no_grad():
        for _ in range(n_batches):
            tokens = get_batch_with_bos(data, batch_size, block_size, device)
            _ = model(tokens, capture_taps=True)
            block2_input = model.block1.last_block_out
            x = model.block2.ln_1(block2_input)  # [B, T, d]
            Bx = (B_torch @ x.transpose(-1, -2)).transpose(-1, -2)  # [B, T, d]

            bos_sum += Bx[:, 0, :].sum(dim=0)
            others_sum += Bx[:, 1:, :].reshape(-1, d).sum(dim=0)
            n_bos += Bx.shape[0]
            n_others += Bx.shape[0] * (Bx.shape[1] - 1)

    bos_dir = (bos_sum / n_bos).cpu().numpy()
    others_dir = (others_sum / n_others).cpu().numpy()
    return bos_dir, others_dir


def cos_sim(a, b):
    return 1 - cosine(a, b)


def analyze_model(checkpoint_path, data, device="cuda", n_batches=50, batch_size=32):
    model, config = load_model(checkpoint_path, device)
    B_torch = get_write_map(model)
    B_np = B_torch.cpu().numpy()

    U, S, Vt = np.linalg.svd(B_np, full_matrices=True)

    bos_dir, others_dir = compute_bos_others_directions(
        model,
        data,
        B_torch,
        n_batches=n_batches,
        batch_size=batch_size,
        block_size=config.block_size,
        device=device,
    )

    del model
    torch.cuda.empty_cache()

    # Contrast direction: what distinguishes BOS from others
    contrast = bos_dir - others_dir
    # Shared direction: mean of both
    shared = (bos_dir + others_dir) / 2.0

    s_tail_mean = np.mean(S[2:50])

    results = {
        "cos_bos_others": cos_sim(bos_dir, others_dir),
        "norm_bos": float(np.linalg.norm(bos_dir)),
        "norm_others": float(np.linalg.norm(others_dir)),
        "norm_contrast": float(np.linalg.norm(contrast)),
        "cos_u1_shared": cos_sim(U[:, 0], shared),
        "cos_u1_contrast": cos_sim(U[:, 0], contrast),
        "cos_u2_shared": cos_sim(U[:, 1], shared),
        "cos_u2_contrast": cos_sim(U[:, 1], contrast),
        "cos_u1_bos": cos_sim(U[:, 0], bos_dir),
        "cos_u1_others": cos_sim(U[:, 0], others_dir),
        "cos_u2_bos": cos_sim(U[:, 1], bos_dir),
        "cos_u2_others": cos_sim(U[:, 1], others_dir),
        "s1": float(S[0]),
        "s2": float(S[1]),
        "s1_ratio": float(S[0] / s_tail_mean),
        "s2_ratio": float(S[1] / s_tail_mean),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r2_1head_ckpt",
        type=str,
        default="nanoGPT/out-2layer-mechanism-r2-1head-postattn/R2/uv1hq205/best_ckpt.pt",
    )
    parser.add_argument(
        "--r0_12head_ckpt",
        type=str,
        default="nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt",
    )
    parser.add_argument("--data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    data = load_owt_data(args.data_dir)

    print("Analyzing R2-1head...")
    r2 = analyze_model(
        args.r2_1head_ckpt, data, args.device, args.n_batches, args.batch_size
    )

    print("Analyzing R0-12head...")
    r0 = analyze_model(
        args.r0_12head_ckpt, data, args.device, args.n_batches, args.batch_size
    )

    # Print full results
    print("\n" + "=" * 80)
    for name, r in [("R2-1head", r2), ("R0-12head", r0)]:
        print(f"\n{name}:")
        for k, v in r.items():
            print(f"  {k}: {v:.4f}")

    # Print LaTeX table
    print("\n" + "=" * 80)
    print("LaTeX Table:")
    print("=" * 80)
    print(r"""
\begin{table}[t]
\centering
\small
\caption{\textbf{SVD alignment of the write map $B = W_O W_V$.}
The BOS and others directions each align with a different top singular vector,
confirming the geometric clock operates as a rotation in the 2D subspace spanned by $u_1, u_2$.
Both top singular values dominate the spectrum ($\sigma/\bar\sigma$ = ratio to mean of $\sigma_{3{:}50}$).}
\label{tab:svd_alignment}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l c cc cc cc}
\toprule
& & \multicolumn{2}{c}{$\cos(u_1, \cdot)$} & \multicolumn{2}{c}{$\cos(u_2, \cdot)$} & \multicolumn{2}{c}{Dominance} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}
Model & $\cos(\text{BOS}, \text{oth.})$ & BOS & others & BOS & others & $\sigma_1/\bar\sigma$ & $\sigma_2/\bar\sigma$ \\
\midrule""")

    for name, r in [("R2-1head", r2), ("R0-12head", r0)]:
        print(
            f"{name} & {r['cos_bos_others']:.2f} "
            f"& {r['cos_u1_bos']:.2f} & {r['cos_u1_others']:.2f} "
            f"& {r['cos_u2_bos']:.2f} & {r['cos_u2_others']:.2f} "
            f"& {r['s1_ratio']:.1f}$\\times$ & {r['s2_ratio']:.1f}$\\times$ \\\\"
        )

    print(r"""\bottomrule
\end{tabular}
\vspace{-2mm}
\end{table}""")


if __name__ == "__main__":
    main()
