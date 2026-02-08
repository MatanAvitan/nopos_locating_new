"""Quantify BOS-anchor decodability after Block-1 attention + succeeding LayerNorm.

This experiment tests Step 1 directly at
    x_i^(B1-attnLN) = LN(\tilde h_i^(1))
where \tilde h_i^(1) is the Block-1 post-attention residual.

For each model, we:
1) Estimate d_BOS^(B1-LN) from x_0 and d_nonBOS^(B1-LN) from mean_{j>0} x_j.
2) Project each token representation onto a chosen direction d:
       s_i = <x_i, d>
3) Fit a 1D affine decoder with learnable scale+bias (a, b):
       y_hat_i = a * s_i + b
4) Evaluate absolute-position decoding on held-out data.

We report metrics for three directions:
- bos_b1ln (primary)
- nonbos_b1ln (control)
- random (control)

We also evaluate a fixed-form decoder in Block-1 write space that avoids using m directly:
    y_hat_m = 1 / (<B^(1) x_m^(B1-attnLN), d_BOS^(B1-write)>^2 + eps) + b
where only b is fitted on training data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))

from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel


BOS_TOKEN_ID = 50256


def load_model(checkpoint_path: Path, device: str) -> TwoLayerMechanismModel:
    """Load a TwoLayerMechanismModel checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint.get("config", checkpoint.get("model_args", {}))
    valid_keys = set(TwoLayerMechanismConfig.__dataclass_fields__.keys())
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = TwoLayerMechanismConfig(**filtered_config)

    model = TwoLayerMechanismModel(config)
    state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_data(data_dir: Path) -> np.memmap:
    """Load OpenWebText validation split."""
    return np.memmap(str(data_dir / "val.bin"), dtype=np.uint16, mode="r")


def get_batch_with_bos(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    device: str,
) -> torch.Tensor:
    """Sample a batch with BOS fixed at position 0."""
    ix = torch.randint(len(data) - (block_size - 1), (batch_size,))
    sequences = []
    for i in ix:
        tail = data[i : i + block_size - 1].astype(np.int64)
        seq = np.concatenate([[BOS_TOKEN_ID], tail])
        sequences.append(torch.from_numpy(seq))
    return torch.stack(sequences).to(device)


def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit norm."""
    return vec / (np.linalg.norm(vec) + 1e-12)


def extract_step1_activations(
    model: TwoLayerMechanismModel, tokens: torch.Tensor
) -> torch.Tensor:
    """Return Step-1 activations x_i^(B1-attnLN) = block1_ln2 [B, T, D]."""
    with torch.no_grad():
        model(tokens, capture_taps=True)
        taps = model.get_all_taps()
    return taps["block1_ln2"]


def get_block1_write_map(model: TwoLayerMechanismModel) -> torch.Tensor:
    """Return Block-1 write map B^(1) = W_O^(1) W_V^(1)."""
    d_model = model.config.n_embd
    w_v = model.block1.attn.c_attn.weight[2 * d_model :, :]
    w_o = model.block1.attn.c_proj.weight
    return w_o @ w_v


def estimate_bos_write_direction(
    model: TwoLayerMechanismModel,
    data: np.memmap,
    block1_write_map: torch.Tensor,
    n_batches: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Estimate d_BOS^(B1-write) from mean BOS write in Block-1 write space."""
    d_model = model.config.n_embd
    block_size = model.config.block_size

    bos_sum = np.zeros(d_model, dtype=np.float64)
    bos_count = 0

    for _ in range(n_batches):
        tokens = get_batch_with_bos(data, batch_size, block_size, device)
        x = extract_step1_activations(model, tokens)
        bx = torch.matmul(x, block1_write_map.T).detach().cpu().numpy()
        bos_sum += bx[:, 0, :].sum(axis=0)
        bos_count += bx.shape[0]

    w_bos = bos_sum / max(1, bos_count)
    return normalize(w_bos.astype(np.float32))


def collect_inverse_square_scores(
    model: TwoLayerMechanismModel,
    data: np.memmap,
    block1_write_map: torch.Tensor,
    d_bos_b1write: np.ndarray,
    n_batches: int,
    batch_size: int,
    device: str,
    inverse_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect inverse-square scores: 1 / (<B^(1)x_m, d>^2 + eps)."""
    block_size = model.config.block_size
    d_tensor = torch.from_numpy(d_bos_b1write).to(device=device, dtype=torch.float32)

    scores = []
    y_values = []
    proj_mean_sum = np.zeros(block_size, dtype=np.float64)
    score_mean_sum = np.zeros(block_size, dtype=np.float64)

    positions_zero_based = np.arange(block_size, dtype=np.float32)

    for _ in range(n_batches):
        tokens = get_batch_with_bos(data, batch_size, block_size, device)
        x = extract_step1_activations(model, tokens)
        bx = torch.matmul(x, block1_write_map.T)

        proj = torch.matmul(bx, d_tensor).detach().cpu().numpy()  # [B, T]
        scaled = 1.0 / (proj**2 + inverse_eps)

        scores.append(scaled.reshape(-1))
        y_values.append(np.tile(positions_zero_based, batch_size))

        proj_mean_sum += proj.mean(axis=0)
        score_mean_sum += scaled.mean(axis=0)

    all_scores = np.concatenate(scores, axis=0)
    y = np.concatenate(y_values, axis=0)
    proj_mean = (proj_mean_sum / n_batches).astype(np.float32)
    score_mean = (score_mean_sum / n_batches).astype(np.float32)
    return all_scores, y, proj_mean, score_mean


def estimate_stage1_directions(
    model: TwoLayerMechanismModel,
    data: np.memmap,
    n_batches: int,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Estimate BOS/non-BOS directions at x_i^(B1-attnLN) = LN(\tilde h_i^(1))."""
    d_model = model.config.n_embd
    block_size = model.config.block_size

    bos_sum = np.zeros(d_model, dtype=np.float64)
    nonbos_sum = np.zeros(d_model, dtype=np.float64)
    bos_count = 0
    nonbos_count = 0

    for _ in range(n_batches):
        tokens = get_batch_with_bos(data, batch_size, block_size, device)
        x = extract_step1_activations(model, tokens).detach().cpu().numpy()  # [B, T, D]

        bos_sum += x[:, 0, :].sum(axis=0)
        nonbos_sum += x[:, 1:, :].sum(axis=(0, 1))
        bos_count += x.shape[0]
        nonbos_count += x.shape[0] * (x.shape[1] - 1)

    u_bos = bos_sum / max(1, bos_count)
    u_nonbos = nonbos_sum / max(1, nonbos_count)

    return {
        "bos_b1ln": normalize(u_bos.astype(np.float32)),
        "nonbos_b1ln": normalize(u_nonbos.astype(np.float32)),
    }


def collect_multidirection_features(
    model: TwoLayerMechanismModel,
    data: np.memmap,
    directions: dict[str, np.ndarray],
    n_batches: int,
    batch_size: int,
    device: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Collect scalar projections s_i=<x_i,d> for multiple directions in one pass."""
    block_size = model.config.block_size
    direction_names = list(directions.keys())
    direction_matrix = np.stack(
        [directions[k] for k in direction_names], axis=1
    )  # [D, K]
    direction_tensor = torch.from_numpy(direction_matrix).to(
        device=device, dtype=torch.float32
    )

    proj_values = {name: [] for name in direction_names}
    mean_proj_sums = {
        name: np.zeros(block_size, dtype=np.float64) for name in direction_names
    }
    y_values = []

    positions = np.arange(block_size, dtype=np.float32)

    for _ in range(n_batches):
        tokens = get_batch_with_bos(data, batch_size, block_size, device)
        x = extract_step1_activations(model, tokens)  # [B, T, D]
        projections = torch.matmul(x, direction_tensor)  # [B, T, K]
        projections_np = projections.detach().cpu().numpy()

        for idx, name in enumerate(direction_names):
            p = projections_np[:, :, idx]
            proj_values[name].append(p.reshape(-1))
            mean_proj_sums[name] += p.mean(axis=0)

        y_batch = np.tile(positions, batch_size)
        y_values.append(y_batch)

    for name in direction_names:
        proj_values[name] = np.concatenate(proj_values[name], axis=0)
        mean_proj_sums[name] = (mean_proj_sums[name] / n_batches).astype(np.float32)

    y = np.concatenate(y_values, axis=0)
    return proj_values, y, mean_proj_sums


def collect_full_activations(
    model: TwoLayerMechanismModel,
    data: np.memmap,
    n_batches: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect full activation vectors at block1_ln2 for all positions.

    Returns:
        X: [N, D] array of activations (float32)
        y: [N] array of position indices (float32)
    """
    block_size = model.config.block_size
    positions = np.arange(block_size, dtype=np.float32)

    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []

    for _ in range(n_batches):
        tokens = get_batch_with_bos(data, batch_size, block_size, device)
        x = extract_step1_activations(model, tokens).detach().cpu().numpy()  # [B, T, D]
        B, T, D = x.shape
        X_chunks.append(x.reshape(B * T, D))
        y_chunks.append(np.tile(positions, B))

    return np.concatenate(X_chunks, axis=0), np.concatenate(y_chunks, axis=0)


def fit_full_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit a ridge regression probe (d->1) and evaluate on test data."""
    probe = Ridge(alpha=alpha, fit_intercept=True)
    probe.fit(X_train, y_train)
    pred_train = probe.predict(X_train)
    pred_test = probe.predict(X_test)

    train_metrics = compute_metrics(y_train, pred_train)
    test_metrics = compute_metrics(y_test, pred_test)

    w = probe.coef_.astype(np.float32)
    w_norm = float(np.linalg.norm(w))
    w_dir = w / (w_norm + 1e-12)

    # Random-label baseline (same probe class, shuffled y on train)
    rng = np.random.default_rng(seed)
    y_train_shuffled = rng.permutation(y_train)
    probe_rand = Ridge(alpha=alpha, fit_intercept=True)
    probe_rand.fit(X_train, y_train_shuffled)
    pred_test_rand = probe_rand.predict(X_test)
    random_label_test_metrics = compute_metrics(y_test, pred_test_rand)

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "random_label_test_metrics": random_label_test_metrics,
        "ridge_alpha": alpha,
        "weight_norm": w_norm,
        "bias": float(probe.intercept_),
        "learned_direction": w_dir.tolist(),
    }


def fit_affine_decoder(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y_hat = a*s + b via least squares."""
    X = np.column_stack([s, np.ones_like(s)])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression and correlation metrics."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot)
    r = float(pearsonr(y_true, y_pred)[0])
    rho = float(spearmanr(y_true, y_pred).correlation)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {
        "r2": r2,
        "pearson_r": r,
        "spearman_rho": rho,
        "mae": mae,
    }


def run_probe_for_model(
    model_name: str,
    model: TwoLayerMechanismModel,
    data: np.memmap,
    train_batches: int,
    test_batches: int,
    batch_size: int,
    seed: int,
    device: str,
    inverse_eps: float,
) -> dict[str, Any]:
    """Run BOS-anchor decodability probe for one model."""
    np.random.seed(seed)

    stage1_dirs = estimate_stage1_directions(
        model=model,
        data=data,
        n_batches=train_batches,
        batch_size=batch_size,
        device=device,
    )

    rng = np.random.default_rng(seed)
    random_dir = normalize(rng.standard_normal(model.config.n_embd).astype(np.float32))

    directions = {
        "bos_b1ln": stage1_dirs["bos_b1ln"],
        "nonbos_b1ln": stage1_dirs["nonbos_b1ln"],
        "random": random_dir,
    }

    train_proj, y_train, train_mean_by_pos = collect_multidirection_features(
        model=model,
        data=data,
        directions=directions,
        n_batches=train_batches,
        batch_size=batch_size,
        device=device,
    )
    test_proj, y_test, test_mean_by_pos = collect_multidirection_features(
        model=model,
        data=data,
        directions=directions,
        n_batches=test_batches,
        batch_size=batch_size,
        device=device,
    )

    direction_metrics = {}
    block_size = model.config.block_size
    for name in directions:
        a, b = fit_affine_decoder(train_proj[name], y_train)
        pred_test = a * test_proj[name] + b
        metrics = compute_metrics(y_test, pred_test)

        # Compute per-position std from test data
        std_by_pos = np.zeros(block_size, dtype=np.float64)
        for pos in range(block_size):
            mask = y_test == pos
            if mask.sum() > 1:
                std_by_pos[pos] = float(np.std(test_proj[name][mask], ddof=1))
        metrics.update(
            {
                "a": a,
                "b": b,
                "mean_projection_by_position_train": train_mean_by_pos[name].tolist(),
                "mean_projection_by_position_test": test_mean_by_pos[name].tolist(),
                "std_projection_by_position_test": std_by_pos.tolist(),
            }
        )
        direction_metrics[name] = metrics

    # Full linear probe (ridge regression, d -> 1)
    print(f"  Collecting full activations for linear probe...")
    X_train_full, y_train_full = collect_full_activations(
        model=model,
        data=data,
        n_batches=train_batches,
        batch_size=batch_size,
        device=device,
    )
    X_test_full, y_test_full = collect_full_activations(
        model=model,
        data=data,
        n_batches=test_batches,
        batch_size=batch_size,
        device=device,
    )
    print(
        f"  Fitting ridge probe on {X_train_full.shape[0]} train, {X_test_full.shape[0]} test samples..."
    )
    full_probe_results = fit_full_linear_probe(
        X_train_full, y_train_full, X_test_full, y_test_full, alpha=1.0, seed=seed
    )

    # Compute cosine similarity between learned probe direction and BOS/nonBOS directions
    learned_dir = np.array(full_probe_results["learned_direction"], dtype=np.float32)
    cos_with_bos = float(np.dot(learned_dir, stage1_dirs["bos_b1ln"]))
    cos_with_nonbos = float(np.dot(learned_dir, stage1_dirs["nonbos_b1ln"]))
    full_probe_results["cos_with_bos_b1ln"] = cos_with_bos
    full_probe_results["cos_with_nonbos_b1ln"] = cos_with_nonbos
    print(
        f"  Full probe test R^2={full_probe_results['test_metrics']['r2']:.4f}, "
        f"cos(w, d_BOS)={cos_with_bos:.4f}, cos(w, d_nonBOS)={cos_with_nonbos:.4f}"
    )

    block1_write_map = get_block1_write_map(model).to(device)
    d_bos_b1write = estimate_bos_write_direction(
        model=model,
        data=data,
        block1_write_map=block1_write_map,
        n_batches=train_batches,
        batch_size=batch_size,
        device=device,
    )

    (
        train_formula_scores,
        y_formula_train,
        train_formula_proj_mean,
        train_formula_score_mean,
    ) = collect_inverse_square_scores(
        model=model,
        data=data,
        block1_write_map=block1_write_map,
        d_bos_b1write=d_bos_b1write,
        n_batches=train_batches,
        batch_size=batch_size,
        device=device,
        inverse_eps=inverse_eps,
    )
    (
        test_formula_scores,
        y_formula_test,
        test_formula_proj_mean,
        test_formula_score_mean,
    ) = collect_inverse_square_scores(
        model=model,
        data=data,
        block1_write_map=block1_write_map,
        d_bos_b1write=d_bos_b1write,
        n_batches=test_batches,
        batch_size=batch_size,
        device=device,
        inverse_eps=inverse_eps,
    )

    b_formula = float(np.mean(y_formula_train - train_formula_scores))
    pred_formula_test = test_formula_scores + b_formula
    appendix_formula_metrics = compute_metrics(y_formula_test, pred_formula_test)
    appendix_formula_metrics.update(
        {
            "b": b_formula,
            "inverse_eps": inverse_eps,
            "decoder": "y_hat = 1 / (<B1 x_m, d_BOS^(B1-write)>^2 + eps) + b",
            "mean_projection_by_position_train": train_formula_proj_mean.tolist(),
            "mean_projection_by_position_test": test_formula_proj_mean.tolist(),
            "mean_inverse_score_by_position_train": train_formula_score_mean.tolist(),
            "mean_inverse_score_by_position_test": test_formula_score_mean.tolist(),
        }
    )

    return {
        "model_name": model_name,
        "block_size": model.config.block_size,
        "n_embd": model.config.n_embd,
        "train_batches": train_batches,
        "test_batches": test_batches,
        "batch_size": batch_size,
        "metrics": direction_metrics,
        "full_linear_probe": full_probe_results,
        "inverse_square_bias_only": appendix_formula_metrics,
    }


def plot_results(results: dict[str, Any], save_path: Path) -> None:
    """Create summary figure for BOS-anchor decodability."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    model_keys = ["attn2_1h", "full_12h"]
    model_labels = [results[k]["model_name"] for k in model_keys]
    colors = ["#0072B2", "#D55E00"]

    # Panel A: mean BOS projection vs position with +/-1 std band
    for model_key, color in zip(model_keys, colors):
        bos_data = results[model_key]["metrics"]["bos_b1ln"]
        mean_proj = np.array(bos_data["mean_projection_by_position_test"])
        std_proj = np.array(bos_data["std_projection_by_position_test"])
        positions = np.arange(mean_proj.shape[0])
        axes[0].plot(
            positions,
            mean_proj,
            linewidth=1.4,
            color=color,
            label=results[model_key]["model_name"],
        )
        axes[0].fill_between(
            positions,
            mean_proj - std_proj,
            mean_proj + std_proj,
            alpha=0.15,
            color=color,
        )
    axes[0].set_xlabel("Position $i$")
    axes[0].set_ylabel(
        r"$\langle x_i^{\mathrm{B1\text{-}LN2}},\; d_{\mathrm{BOS}}\rangle$"
    )
    axes[0].set_title(r"(a) BOS projection after LN2")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=7)

    # Panel B: mean nonBOS projection vs position with +/-1 std band
    for model_key, color in zip(model_keys, colors):
        nonbos_data = results[model_key]["metrics"]["nonbos_b1ln"]
        mean_proj = np.array(nonbos_data["mean_projection_by_position_test"])
        std_proj = np.array(nonbos_data["std_projection_by_position_test"])
        positions = np.arange(mean_proj.shape[0])
        axes[1].plot(
            positions,
            mean_proj,
            linewidth=1.4,
            color=color,
            label=results[model_key]["model_name"],
        )
        axes[1].fill_between(
            positions,
            mean_proj - std_proj,
            mean_proj + std_proj,
            alpha=0.15,
            color=color,
        )
    axes[1].set_xlabel("Position $i$")
    axes[1].set_ylabel(
        r"$\langle x_i^{\mathrm{B1\text{-}LN2}},\; d_{\mathrm{nonBOS}}\rangle$"
    )
    axes[1].set_title(r"(b) nonBOS projection after LN2")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="lower right", fontsize=7)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def to_serializable(obj: Any) -> Any:
    """Convert numpy scalars/arrays recursively for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attn2_ckpt",
        type=str,
        default="model_backups/R2_ATTN2-1H/best_ckpt.pt",
    )
    parser.add_argument(
        "--full12h_ckpt",
        type=str,
        default="model_backups/R0_FULL-12H/best_ckpt.pt",
    )
    parser.add_argument("--data_dir", type=str, default="nanoGPT/data/openwebtext")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/bos_anchor_decodability_after_ln",
    )
    parser.add_argument(
        "--paper_dir",
        type=str,
        default="overleaf/nopos_icml_2026/plots",
    )
    parser.add_argument("--train_batches", type=int, default=20)
    parser.add_argument("--test_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--inverse_eps", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    save_dir = ROOT_DIR / args.save_dir
    paper_dir = ROOT_DIR / args.paper_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(ROOT_DIR / args.data_dir)

    attn2_model = load_model(ROOT_DIR / args.attn2_ckpt, device)
    full_model = load_model(ROOT_DIR / args.full12h_ckpt, device)

    print("Running BOS-anchor probe for ATTN2-1H...")
    results_attn2 = run_probe_for_model(
        model_name="ATTN2-1H",
        model=attn2_model,
        data=data,
        train_batches=args.train_batches,
        test_batches=args.test_batches,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        inverse_eps=args.inverse_eps,
    )

    print("Running BOS-anchor probe for FULL-12H...")
    results_full = run_probe_for_model(
        model_name="FULL-12H",
        model=full_model,
        data=data,
        train_batches=args.train_batches,
        test_batches=args.test_batches,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        inverse_eps=args.inverse_eps,
    )

    all_results = {
        "attn2_1h": results_attn2,
        "full_12h": results_full,
    }

    json_path = save_dir / "bos_anchor_decodability_results.json"
    with open(json_path, "w") as f:
        json.dump(to_serializable(all_results), f, indent=2)

    fig_path = save_dir / "bos_anchor_decodability_after_ln.pdf"
    plot_results(all_results, fig_path)

    shutil.copy2(fig_path, paper_dir / fig_path.name)
    png_path = fig_path.with_suffix(".png")
    if png_path.exists():
        shutil.copy2(png_path, paper_dir / png_path.name)

    print("Saved:")
    print(f"  {json_path}")
    print(f"  {fig_path}")
    print(f"  {paper_dir / fig_path.name}")

    for model_key in ["attn2_1h", "full_12h"]:
        model_name = all_results[model_key]["model_name"]
        primary = all_results[model_key]["metrics"]["bos_b1ln"]
        nonbos = all_results[model_key]["metrics"]["nonbos_b1ln"]
        full_probe = all_results[model_key]["full_linear_probe"]
        inverse_square = all_results[model_key]["inverse_square_bias_only"]
        print(f"\n--- {model_name} ---")
        print(
            f"  BOS 1D decoder:      R^2={primary['r2']:.4f}, "
            f"pearson={primary['pearson_r']:.4f}, spearman={primary['spearman_rho']:.4f}"
        )
        print(
            f"  nonBOS 1D decoder:   R^2={nonbos['r2']:.4f}, "
            f"pearson={nonbos['pearson_r']:.4f}, spearman={nonbos['spearman_rho']:.4f}"
        )
        print(
            f"  Full linear probe:   R^2={full_probe['test_metrics']['r2']:.4f}, "
            f"pearson={full_probe['test_metrics']['pearson_r']:.4f}, "
            f"spearman={full_probe['test_metrics']['spearman_rho']:.4f}"
        )
        print(
            f"  Full probe random-label baseline: "
            f"R^2={full_probe['random_label_test_metrics']['r2']:.4f}, "
            f"pearson={full_probe['random_label_test_metrics']['pearson_r']:.4f}, "
            f"spearman={full_probe['random_label_test_metrics']['spearman_rho']:.4f}"
        )
        print(
            f"    cos(w, d_BOS)={full_probe['cos_with_bos_b1ln']:.4f}, "
            f"cos(w, d_nonBOS)={full_probe['cos_with_nonbos_b1ln']:.4f}"
        )
        print(
            f"  Inverse-square bias: R^2={inverse_square['r2']:.4f}, "
            f"pearson={inverse_square['pearson_r']:.4f}, "
            f"spearman={inverse_square['spearman_rho']:.4f}"
        )


if __name__ == "__main__":
    main()
