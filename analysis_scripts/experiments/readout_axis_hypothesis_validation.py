"""Validate readout-axis hypothesis on two representation spaces.

Spaces:
1) attn_centered: o_i - b_attn  (clean Step-4 attention geometry)
2) head_input:    h_i^(2)       (actual input to learned linear head)

For each checkpoint, compares delta-axis vs learned-head axis on:
- Monotonicity (Spearman)
- Linear decodability (R^2)
- SNR (between-position variance / within-position variance)
- Endpoint separation
- Residual sensitivity (available in attn_centered space)
- Optional rotation sweep in span{delta_hat, head_hat}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))

from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel


BOS_TOKEN_ID = 50256
EPS = 1e-12
SPACE_ATTN = "attn_centered"
SPACE_HEAD = "head_input"


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    rho = spearmanr(x, y).correlation
    if rho is None or np.isnan(rho):
        return 0.0
    return float(rho)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-14 or sy < 1e-14:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a, dim=0, eps=1e-12)
    b = F.normalize(b, dim=0, eps=1e-12)
    return float(torch.dot(a, b).item())


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def load_model(checkpoint_path: Path, device: str) -> TwoLayerMechanismModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    config = TwoLayerMechanismConfig(
        block_size=cfg["block_size"],
        vocab_size=cfg["vocab_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        dropout=0.0,
        norm_type=cfg["norm_type"],
        bias=True,
        use_regression=True,
    )
    model = TwoLayerMechanismModel(config)
    state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def load_data(data_dir: Path) -> np.memmap:
    return np.memmap(str(data_dir / "train.bin"), dtype=np.uint16, mode="r")


def sample_tokens(
    data: np.memmap, seq_len: int, batch_size: int, device: str
) -> torch.Tensor:
    ix = torch.randint(len(data) - (seq_len - 1), (batch_size,))
    tokens = []
    for i in ix:
        seq = np.concatenate(
            [[BOS_TOKEN_ID], data[i : i + seq_len - 1].astype(np.int64)]
        )
        tokens.append(torch.from_numpy(seq))
    return torch.stack(tokens).to(device)


def default_sampling(seq_len: int) -> tuple[int, int]:
    if seq_len <= 128:
        return 16, 8
    if seq_len <= 1024:
        return 8, 8
    return 1, 4


def _alloc_signal_buffer(n_total: int, seq_len: int) -> dict[str, np.ndarray]:
    return {
        "delta": np.full((n_total, seq_len), np.nan, dtype=np.float64),
        "head": np.full((n_total, seq_len), np.nan, dtype=np.float64),
        "perp": np.full((n_total, seq_len), np.nan, dtype=np.float64),
    }


def analyze_checkpoint(
    model: TwoLayerMechanismModel,
    data: np.memmap,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    device: str,
    theta_steps: int,
) -> dict[str, Any]:
    d_model = model.config.n_embd
    W_V = model.block2.attn.c_attn.weight[2 * d_model :, :].detach()
    W_O = model.block2.attn.c_proj.weight.detach()
    b_O = model.block2.attn.c_proj.bias.detach()

    # Pass 1: estimate w_BOS and w_nonBOS.
    bos_sum = torch.zeros(d_model, device=device)
    non_sum = torch.zeros(d_model, device=device)
    bos_count = 0
    non_count = 0

    with torch.no_grad():
        for _ in range(n_batches):
            tokens = sample_tokens(data, seq_len, batch_size, device)
            model(tokens, capture_taps=True)
            ln2_1 = model.get_all_taps()["block2_ln1"]
            Wo_v = (ln2_1 @ W_V.T) @ W_O.T

            bos_sum += Wo_v[:, 0, :].sum(dim=0)
            bos_count += Wo_v.shape[0]

            non = Wo_v[:, 1:, :]
            non_sum += non.sum(dim=(0, 1))
            non_count += non.shape[0] * non.shape[1]

    w_bos = bos_sum / max(bos_count, 1)
    w_non = non_sum / max(non_count, 1)
    delta = w_non - w_bos

    if not hasattr(model, "pos_head") or model.pos_head is None:
        raise RuntimeError("Model has no regression head (pos_head)")
    w_head = model.pos_head.weight.detach().squeeze().to(device)

    delta_hat = F.normalize(delta, dim=0, eps=1e-12)
    head_hat = F.normalize(w_head, dim=0, eps=1e-12)
    d_non_hat = F.normalize(w_non, dim=0, eps=1e-12)
    d_bos_hat = F.normalize(w_bos, dim=0, eps=1e-12)

    # Build orthonormal basis for rotation sweep.
    proj = torch.dot(head_hat, delta_hat)
    v_perp = head_hat - proj * delta_hat
    v_perp_norm = torch.norm(v_perp).item()
    if v_perp_norm < 1e-8:
        v_perp = torch.zeros_like(delta_hat)
    else:
        v_perp = v_perp / v_perp_norm

    n_total = batch_size * n_batches
    positions = np.arange(seq_len, dtype=np.float64)

    signals = {
        SPACE_ATTN: _alloc_signal_buffer(n_total, seq_len),
        SPACE_HEAD: _alloc_signal_buffer(n_total, seq_len),
    }

    # eta/non projections are defined from Step-4 decomposition in attention space.
    eta_proj = _alloc_signal_buffer(n_total, seq_len)
    non_proj = _alloc_signal_buffer(n_total, seq_len)

    row = 0
    with torch.no_grad():
        for _ in range(n_batches):
            tokens = sample_tokens(data, seq_len, batch_size, device)
            model(tokens, capture_taps=True)
            taps = model.get_all_taps()
            _, attn2 = model.get_attention_weights()

            ln2_1 = taps["block2_ln1"]
            attn_out = taps["block2_attn"]
            block2_out = taps["block2_out"]

            Wo_v = (ln2_1 @ W_V.T) @ W_O.T
            attn_no_bias = attn_out - b_O
            head_input = model.ln_f(block2_out)

            alpha = attn2.mean(dim=1)
            alpha_bos = alpha[:, :, 0]
            alpha_non = alpha[:, :, 1:]
            non_mass = 1.0 - alpha_bos

            non_term = torch.einsum("btk,bkd->btd", alpha_non, Wo_v[:, 1:, :])
            eta = non_term - non_mass.unsqueeze(-1) * w_non

            valid = non_mass > 1e-6
            valid[:, 0] = False
            valid_np = valid.detach().cpu().numpy()

            reps = {
                SPACE_ATTN: attn_no_bias,
                SPACE_HEAD: head_input,
            }
            for space_name, rep in reps.items():
                sd = torch.einsum("btd,d->bt", rep, delta_hat)
                sh = torch.einsum("btd,d->bt", rep, head_hat)
                sp = torch.einsum("btd,d->bt", rep, v_perp)
                for key, src in [("delta", sd), ("head", sh), ("perp", sp)]:
                    arr = src.detach().cpu().numpy().astype(np.float64)
                    arr[~valid_np] = np.nan
                    signals[space_name][key][row : row + batch_size, :] = arr

            ed = torch.einsum("btd,d->bt", eta, delta_hat)
            eh = torch.einsum("btd,d->bt", eta, head_hat)
            ep = torch.einsum("btd,d->bt", eta, v_perp)

            nd = torch.einsum("btd,d->bt", non_term, delta_hat)
            nh = torch.einsum("btd,d->bt", non_term, head_hat)
            np_ = torch.einsum("btd,d->bt", non_term, v_perp)

            for key, src in [("delta", ed), ("head", eh), ("perp", ep)]:
                arr = src.detach().cpu().numpy().astype(np.float64)
                arr[~valid_np] = np.nan
                eta_proj[key][row : row + batch_size, :] = arr

            for key, src in [("delta", nd), ("head", nh), ("perp", np_)]:
                arr = src.detach().cpu().numpy().astype(np.float64)
                arr[~valid_np] = np.nan
                non_proj[key][row : row + batch_size, :] = arr

            row += batch_size

    def axis_metrics(
        signal: np.ndarray,
        eta_axis: np.ndarray | None = None,
        non_axis: np.ndarray | None = None,
    ) -> dict[str, Any]:
        mean_curve = np.nanmean(signal, axis=0)
        spearman_mean_curve = _safe_spearman(mean_curve[1:], positions[1:])

        flat_signal = signal[:, 1:].reshape(-1)
        flat_pos = np.tile(positions[1:], signal.shape[0])
        valid = ~np.isnan(flat_signal)
        flat_signal = flat_signal[valid]
        flat_pos = flat_pos[valid]

        spearman_token = _safe_spearman(flat_signal, flat_pos)
        corr = _safe_corr(flat_signal, flat_pos)
        r2_linear = corr * corr

        between = float(np.nanvar(mean_curve[1:], ddof=0))
        within = float(np.nanmean(np.nanvar(signal[:, 1:], axis=0, ddof=0)))
        snr = float(between / (within + EPS))

        if eta_axis is not None and non_axis is not None:
            abs_eta = np.abs(eta_axis)
            ratio = abs_eta / (np.abs(non_axis) + EPS)
            residual_abs_mean = float(np.nanmean(abs_eta))
            residual_abs_median = float(np.nanmedian(abs_eta))
            residual_ratio_mean = float(np.nanmean(ratio))
            residual_ratio_median = float(np.nanmedian(ratio))
        else:
            residual_abs_mean = float("nan")
            residual_abs_median = float("nan")
            residual_ratio_mean = float("nan")
            residual_ratio_median = float("nan")

        return {
            "spearman_mean_curve": spearman_mean_curve,
            "spearman_token": spearman_token,
            "r2_linear": float(r2_linear),
            "snr": snr,
            "between_position_var": between,
            "within_position_var": within,
            "residual_abs_mean": residual_abs_mean,
            "residual_abs_median": residual_abs_median,
            "residual_ratio_mean": residual_ratio_mean,
            "residual_ratio_median": residual_ratio_median,
            "mean_projection_by_pos": mean_curve,
        }

    metrics_by_space: dict[str, dict[str, Any]] = {}
    for space_name in [SPACE_ATTN, SPACE_HEAD]:
        use_residual = space_name == SPACE_ATTN
        metrics_by_space[space_name] = {
            "delta": axis_metrics(
                signals[space_name]["delta"],
                eta_proj["delta"] if use_residual else None,
                non_proj["delta"] if use_residual else None,
            ),
            "head": axis_metrics(
                signals[space_name]["head"],
                eta_proj["head"] if use_residual else None,
                non_proj["head"] if use_residual else None,
            ),
        }

    endpoint = {
        "delta_axis": {
            "bos_dot": float(torch.dot(w_bos, delta_hat).item()),
            "nonbos_dot": float(torch.dot(w_non, delta_hat).item()),
            "gap": float(
                (torch.dot(w_non, delta_hat) - torch.dot(w_bos, delta_hat)).item()
            ),
        },
        "head_axis": {
            "bos_dot": float(torch.dot(w_bos, head_hat).item()),
            "nonbos_dot": float(torch.dot(w_non, head_hat).item()),
            "gap": float(
                (torch.dot(w_non, head_hat) - torch.dot(w_bos, head_hat)).item()
            ),
        },
    }

    theta_deg = np.linspace(-90.0, 90.0, theta_steps, dtype=np.float64)
    rotation_by_space: dict[str, dict[str, Any]] = {}
    for space_name in [SPACE_ATTN, SPACE_HEAD]:
        r2_vals = []
        snr_vals = []
        resid_vals = []

        for th in theta_deg:
            rad = math.radians(float(th))
            c = math.cos(rad)
            s = math.sin(rad)

            sig = c * signals[space_name]["delta"] + s * signals[space_name]["perp"]
            if space_name == SPACE_ATTN:
                eta_u = c * eta_proj["delta"] + s * eta_proj["perp"]
                non_u = c * non_proj["delta"] + s * non_proj["perp"]
                m = axis_metrics(sig, eta_u, non_u)
            else:
                m = axis_metrics(sig)

            r2_vals.append(float(m["r2_linear"]))
            snr_vals.append(float(m["snr"]))
            resid_vals.append(float(m["residual_ratio_median"]))

        r2_arr = np.asarray(r2_vals)
        snr_arr = np.asarray(snr_vals)
        resid_arr = np.asarray(resid_vals)

        rotation_by_space[space_name] = {
            "theta_deg": theta_deg,
            "r2_linear": r2_arr,
            "snr": snr_arr,
            "residual_ratio_median": resid_arr,
            "theta_best_r2": float(theta_deg[int(np.nanargmax(r2_arr))]),
            "theta_best_snr": float(theta_deg[int(np.nanargmax(snr_arr))]),
            "theta_best_residual": float(theta_deg[int(np.nanargmin(resid_arr))])
            if space_name == SPACE_ATTN
            else float("nan"),
            "theta_head_deg": float(
                math.degrees(math.atan2(v_perp_norm, float(proj.item())))
            ),
            "theta_delta_deg": 0.0,
        }

    axis_alignment = {
        "cos_head_delta": _safe_cosine(head_hat, delta_hat),
        "cos_head_nonbos": _safe_cosine(head_hat, d_non_hat),
        "cos_head_bos": _safe_cosine(head_hat, d_bos_hat),
        "cos_nonbos_delta": _safe_cosine(d_non_hat, delta_hat),
        "head_angle_from_delta_deg": float(
            math.degrees(math.atan2(v_perp_norm, float(proj.item())))
        ),
    }

    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "n_sequences": n_total,
        "axis_alignment": axis_alignment,
        "spaces": metrics_by_space,
        "endpoint_separation": endpoint,
        "rotation_sweep": rotation_by_space,
    }


def plot_mean_projection(results: dict[str, Any], save_dir: Path) -> None:
    for name, exp in results.items():
        for space_name, space in exp["spaces"].items():
            delta_curve = np.asarray(
                space["delta"]["mean_projection_by_pos"], dtype=np.float64
            )
            head_curve = np.asarray(
                space["head"]["mean_projection_by_pos"], dtype=np.float64
            )
            pos = np.arange(delta_curve.shape[0])

            fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.4))
            ax.plot(pos, delta_curve, linewidth=1.4, label="Delta axis")
            ax.plot(pos, head_curve, linewidth=1.4, label="Learned head axis")
            ax.set_title(f"Mean projection vs position: {name} ({space_name})")
            ax.set_xlabel("Position")
            ax.set_ylabel("Mean projection")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(
                save_dir / f"mean_projection_{name}_{space_name}.png",
                dpi=220,
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_rotation_sweep(results: dict[str, Any], save_dir: Path) -> None:
    for name, exp in results.items():
        for space_name, rot in exp["rotation_sweep"].items():
            theta = np.asarray(rot["theta_deg"], dtype=np.float64)
            r2 = np.asarray(rot["r2_linear"], dtype=np.float64)
            snr = np.asarray(rot["snr"], dtype=np.float64)

            fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

            ax = axes[0]
            ax.plot(theta, r2, linewidth=1.3)
            ax.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.7)
            ax.axvline(rot["theta_head_deg"], linestyle=":", linewidth=1.0, alpha=0.8)
            ax.set_title(f"R^2 sweep: {name} ({space_name})")
            ax.set_xlabel("Theta (deg)")
            ax.set_ylabel("R^2")
            ax.grid(True, alpha=0.2)

            ax = axes[1]
            ax.plot(theta, snr, linewidth=1.3)
            ax.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.7)
            ax.axvline(rot["theta_head_deg"], linestyle=":", linewidth=1.0, alpha=0.8)
            ax.set_title(f"SNR sweep: {name} ({space_name})")
            ax.set_xlabel("Theta (deg)")
            ax.set_ylabel("SNR")
            ax.grid(True, alpha=0.2)

            fig.tight_layout()
            fig.savefig(
                save_dir / f"rotation_sweep_{name}_{space_name}.png",
                dpi=220,
                bbox_inches="tight",
            )
            plt.close(fig)


def default_experiments() -> dict[str, dict[str, Any]]:
    return {
        "R0_128_FULL-12H": {
            "checkpoint": "nanoGPT/out-2layer-mechanism/R0/best_ckpt.pt",
            "seq_len": 128,
        },
        "R2_128_ATTN2-1H": {
            "checkpoint": "nanoGPT/out-2layer-mechanism-r2-1head-attnonly-fullblock-40k/R2/o4w7v8dv/best_ckpt.pt",
            "seq_len": 128,
        },
        "R0_1024_FULL-12H": {
            "checkpoint": "nanoGPT/out-mechanism-R0-1024/R0/nuacla0w/best_ckpt.pt",
            "seq_len": 1024,
        },
        "R2_1024_ATTN2-1H": {
            "checkpoint": "nanoGPT/out-mechanism-R2-1024/R2/t72g9e8p/best_ckpt.pt",
            "seq_len": 1024,
        },
        "R0_8192_FULL-12H": {
            "checkpoint": "nanoGPT/out-mechanism-R0-8192/R0/lwrgnltx/best_ckpt.pt",
            "seq_len": 8192,
        },
        "R2_8192_12H": {
            "checkpoint": "nanoGPT/out-mechanism-R2-12h-8192/R2/pblr79la/best_ckpt.pt",
            "seq_len": 8192,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument(
        "--save_dir", type=str, default="results/readout_axis_hypothesis_validation"
    )
    parser.add_argument("--theta_steps", type=int, default=121)
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help="Comma-separated subset of experiment names or 'all'.",
    )
    parser.add_argument("--batch_override", type=int, default=0)
    parser.add_argument("--batches_override", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    experiments = default_experiments()
    if args.experiments != "all":
        keep = {x.strip() for x in args.experiments.split(",") if x.strip()}
        experiments = {k: v for k, v in experiments.items() if k in keep}
        if not experiments:
            raise ValueError("No matching experiments selected")

    data = load_data(ROOT_DIR / args.data_dir)
    save_dir = ROOT_DIR / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {
        "device": device,
        "theta_steps": int(args.theta_steps),
        "spaces": [SPACE_ATTN, SPACE_HEAD],
        "experiments": {},
    }

    for name, cfg in experiments.items():
        ckpt = ROOT_DIR / cfg["checkpoint"]
        seq_len = int(cfg["seq_len"])
        if not ckpt.exists():
            print(f"[skip] missing checkpoint: {ckpt}")
            continue

        bs, nb = default_sampling(seq_len)
        if args.batch_override > 0:
            bs = int(args.batch_override)
        if args.batches_override > 0:
            nb = int(args.batches_override)

        print("\n" + "=" * 64)
        print(f"Running {name}")
        print(f"checkpoint={ckpt}")
        print(f"seq_len={seq_len}, batch_size={bs}, n_batches={nb}")
        print("=" * 64)

        model = load_model(ckpt, device)
        result = analyze_checkpoint(
            model=model,
            data=data,
            seq_len=seq_len,
            batch_size=bs,
            n_batches=nb,
            device=device,
            theta_steps=args.theta_steps,
        )
        all_results["experiments"][name] = result

        al = result["axis_alignment"]
        print(f"cos(head,delta)={al['cos_head_delta']:.3f}")
        for space_name in [SPACE_ATTN, SPACE_HEAD]:
            md = result["spaces"][space_name]["delta"]
            mh = result["spaces"][space_name]["head"]
            print(
                f"  {space_name}: "
                f"spearman(delta/head)={md['spearman_mean_curve']:.3f}/{mh['spearman_mean_curve']:.3f}, "
                f"r2(delta/head)={md['r2_linear']:.3f}/{mh['r2_linear']:.3f}, "
                f"snr(delta/head)={md['snr']:.3f}/{mh['snr']:.3f}"
            )

    json_path = save_dir / "readout_axis_validation_results.json"
    with open(json_path, "w") as f:
        json.dump(_to_serializable(all_results), f, indent=2)

    plot_mean_projection(all_results["experiments"], save_dir)
    plot_rotation_sweep(all_results["experiments"], save_dir)

    print("\nSaved:")
    print(f"  {json_path}")
    print(f"  {save_dir}/*.png")


if __name__ == "__main__":
    main()
