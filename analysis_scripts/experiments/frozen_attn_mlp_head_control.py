"""Frozen-attention MLP+head control for ATTN2-1H long context.

Purpose:
1) Start from trained ATTN2-1H checkpoints.
2) Freeze attention and all upstream components.
3) Train only Block2.LN2 + Block2.MLP + Pos Head.
4) Compare against baseline ATTN2-1H and existing from-scratch strict Attn+MLP+Head checkpoints.

This directly tests whether long-context degradation is mainly a readout nonlinearity bottleneck.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT_DIR / "nanoGPT"))
from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel


BOS_TOKEN_ID = 50256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RunSpec:
    name: str
    seq_len: int
    baseline_ckpt: Path
    strict_attn_mlp_ckpt: Path
    finetune_steps: int
    batch_size: int
    grad_accum: int
    lr: float
    eval_every: int
    train_eval_seq: int
    final_eval_seq: int


def load_memmap(split: str) -> np.memmap:
    return np.memmap(
        ROOT_DIR / f"nanoGPT/data/openwebtext/{split}.bin", dtype=np.uint16, mode="r"
    )


def sample_batch(
    data: np.memmap,
    seq_len: int,
    batch_size: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    starts = rng.integers(0, len(data) - (seq_len - 1), size=batch_size)
    arr = np.empty((batch_size, seq_len), dtype=np.int64)
    arr[:, 0] = BOS_TOKEN_ID
    for i, s in enumerate(starts):
        arr[i, 1:] = data[s : s + seq_len - 1].astype(np.int64)
    return torch.from_numpy(arr).to(DEVICE)


def sample_fixed_sequences(
    data: np.memmap,
    seq_len: int,
    n_seq: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - (seq_len - 1), size=n_seq)
    arr = np.empty((n_seq, seq_len), dtype=np.int64)
    arr[:, 0] = BOS_TOKEN_ID
    for i, s in enumerate(starts):
        arr[i, 1:] = data[s : s + seq_len - 1].astype(np.int64)
    return arr


def load_model_from_ckpt(
    ckpt_path: Path,
) -> tuple[TwoLayerMechanismModel, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model_cfg = TwoLayerMechanismConfig(
        block_size=cfg["block_size"],
        vocab_size=cfg["vocab_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        dropout=0.0,
        norm_type=cfg["norm_type"],
        bias=True,
        use_regression=True,
    )
    model = TwoLayerMechanismModel(model_cfg)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.to(DEVICE)
    return model, ckpt


@torch.no_grad()
def evaluate_model(
    model: TwoLayerMechanismModel,
    seqs: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    all_pred = []
    all_true = []
    all_loss = []
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if DEVICE == "cuda"
        else torch.no_grad()
    )

    for s in range(0, seqs.shape[0], batch_size):
        e = min(seqs.shape[0], s + batch_size)
        toks = torch.from_numpy(seqs[s:e]).to(DEVICE)
        targets = torch.arange(
            seqs.shape[1], dtype=torch.float32, device=DEVICE
        ).unsqueeze(0).repeat(e - s, 1) / max(seqs.shape[1] - 1, 1)
        with ctx:
            pred, loss = model(toks, targets=targets, capture_taps=False)
        all_pred.append(pred.squeeze(-1).float().detach().cpu().numpy())
        all_true.append(targets.detach().cpu().numpy())
        all_loss.append(float(loss.item()))

    pred = np.concatenate(all_pred, axis=0).astype(np.float64)
    true = np.concatenate(all_true, axis=0).astype(np.float64)
    y_true = true.reshape(-1)
    y_pred = pred.reshape(-1)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2_cod = 1.0 - ss_res / (ss_tot + 1e-12)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    scale = float(max(seqs.shape[1] - 1, 1))
    return {
        "loss": float(np.mean(all_loss)),
        "r2_cod": float(r2_cod),
        "r2_corr": float(corr * corr),
        "pearson_r": corr,
        "mae_norm": mae,
        "rmse_norm": rmse,
        "mae_abs": float(mae * scale),
        "rmse_abs": float(rmse * scale),
        "n_seq": int(seqs.shape[0]),
    }


def set_frozen_attn_mlp_head_trainable(model: TwoLayerMechanismModel) -> int:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.block2.ln_2.parameters():
        p.requires_grad = True
    for p in model.block2.mlp.parameters():
        p.requires_grad = True
    for p in model.pos_head.parameters():
        p.requires_grad = True
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def finetune_frozen_attention_control(
    model: TwoLayerMechanismModel,
    train_data: np.memmap,
    val_seqs: np.ndarray,
    spec: RunSpec,
    seed: int,
) -> tuple[TwoLayerMechanismModel, dict[str, Any]]:
    n_trainable = set_frozen_attn_mlp_head_trainable(model)
    model.train()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=spec.lr,
        weight_decay=1e-4,
    )
    rng = np.random.default_rng(seed)
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if DEVICE == "cuda"
        else torch.no_grad()
    )

    best_state = None
    best_val = float("inf")
    hist: list[dict[str, float]] = []

    for step in range(1, spec.finetune_steps + 1):
        opt.zero_grad(set_to_none=True)
        train_loss_acc = 0.0

        for _ in range(spec.grad_accum):
            toks = sample_batch(train_data, spec.seq_len, spec.batch_size, rng)
            targets = torch.arange(
                spec.seq_len, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0).repeat(spec.batch_size, 1) / max(spec.seq_len - 1, 1)
            with ctx:
                _, loss = model(toks, targets=targets, capture_taps=False)
                loss = loss / spec.grad_accum
            loss.backward()
            train_loss_acc += float(loss.item())

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()

        if step % spec.eval_every == 0 or step == spec.finetune_steps:
            val_metrics = evaluate_model(
                model, val_seqs, batch_size=max(1, min(8, spec.batch_size))
            )
            rec = {
                "step": step,
                "train_loss": train_loss_acc,
                "val_loss": val_metrics["loss"],
                "val_r2_cod": val_metrics["r2_cod"],
                "val_r2_corr": val_metrics["r2_corr"],
                "val_mae_abs": val_metrics["mae_abs"],
            }
            hist.append(rec)
            print(
                f"[{spec.name}] step {step:5d}/{spec.finetune_steps} "
                f"train_loss={train_loss_acc:.4f} val_r2={val_metrics['r2_corr']:.4f} val_mae_abs={val_metrics['mae_abs']:.2f}",
                flush=True,
            )

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    info = {
        "n_trainable_params": n_trainable,
        "best_val_loss": float(best_val),
        "history": hist,
        "train_cfg": {
            "steps": spec.finetune_steps,
            "batch_size": spec.batch_size,
            "grad_accum": spec.grad_accum,
            "lr": spec.lr,
            "eval_every": spec.eval_every,
        },
    }
    return model, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run shorter preliminary schedule for faster turnaround",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="results/frozen_attn_mlp_head_control/frozen_attn_mlp_head_8192_32768.json",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data = load_memmap("train")
    val_data = load_memmap("val")

    specs = [
        RunSpec(
            name="R2_8192_ATTN2-1H",
            seq_len=8192,
            baseline_ckpt=ROOT_DIR
            / "nanoGPT/out-mechanism-R2-8192/R2/czld6m70/best_ckpt.pt",
            strict_attn_mlp_ckpt=ROOT_DIR
            / "nanoGPT/out-mechanism-R2-8192-attnmlphead-strict-b200/R2/best_ckpt.pt",
            finetune_steps=900,
            batch_size=1,
            grad_accum=8,
            lr=2.0e-4,
            eval_every=75,
            train_eval_seq=64,
            final_eval_seq=256,
        ),
        RunSpec(
            name="R2_32768_ATTN2-1H",
            seq_len=32768,
            baseline_ckpt=ROOT_DIR
            / "nanoGPT/out-mechanism-R2-32k/R2/5r01gjjo/best_ckpt.pt",
            strict_attn_mlp_ckpt=ROOT_DIR
            / "nanoGPT/out-mechanism-R2-32k-attnmlphead-strict-b200/R2/best_ckpt.pt",
            finetune_steps=500,
            batch_size=1,
            grad_accum=4,
            lr=1.5e-4,
            eval_every=50,
            train_eval_seq=32,
            final_eval_seq=96,
        ),
    ]

    if args.quick:
        specs = [
            RunSpec(
                name=s.name,
                seq_len=s.seq_len,
                baseline_ckpt=s.baseline_ckpt,
                strict_attn_mlp_ckpt=s.strict_attn_mlp_ckpt,
                finetune_steps=max(120, s.finetune_steps // 5),
                batch_size=s.batch_size,
                grad_accum=max(2, s.grad_accum // 2),
                lr=s.lr,
                eval_every=max(20, s.eval_every // 2),
                train_eval_seq=max(16, s.train_eval_seq // 2),
                final_eval_seq=max(32, s.final_eval_seq // 3),
            )
            for s in specs
        ]

    out: dict[str, Any] = {
        "device": DEVICE,
        "seed": args.seed,
        "experiments": {},
    }

    for i, spec in enumerate(specs):
        print("\n" + "=" * 80)
        print(f"Running {spec.name} (L={spec.seq_len})")
        print("=" * 80)

        baseline_model, _ = load_model_from_ckpt(spec.baseline_ckpt)
        strict_model, _ = load_model_from_ckpt(spec.strict_attn_mlp_ckpt)

        val_small = sample_fixed_sequences(
            val_data, spec.seq_len, spec.train_eval_seq, args.seed + 10 + i
        )
        val_final = sample_fixed_sequences(
            val_data, spec.seq_len, spec.final_eval_seq, args.seed + 100 + i
        )

        baseline_small = evaluate_model(baseline_model, val_small, batch_size=1)
        baseline_final = evaluate_model(baseline_model, val_final, batch_size=1)
        strict_final = evaluate_model(strict_model, val_final, batch_size=1)

        tuned_model, tune_info = finetune_frozen_attention_control(
            model=baseline_model,
            train_data=train_data,
            val_seqs=val_small,
            spec=spec,
            seed=args.seed + 1000 + i,
        )
        tuned_final = evaluate_model(tuned_model, val_final, batch_size=1)

        out["experiments"][spec.name] = {
            "seq_len": spec.seq_len,
            "checkpoints": {
                "baseline_attn_only": str(spec.baseline_ckpt),
                "strict_attn_mlp_from_scratch": str(spec.strict_attn_mlp_ckpt),
            },
            "baseline_small_eval": baseline_small,
            "baseline_final_eval": baseline_final,
            "strict_attn_mlp_final_eval": strict_final,
            "frozen_attn_mlp_head_finetune": tune_info,
            "frozen_attn_mlp_head_final_eval": tuned_final,
        }

        print(
            f"[{spec.name}] final R2 corr: baseline={baseline_final['r2_corr']:.4f} | "
            f"frozen-attn-mlp-head={tuned_final['r2_corr']:.4f} | "
            f"strict-attn-mlp={strict_final['r2_corr']:.4f}",
            flush=True,
        )

    save_path = ROOT_DIR / args.save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to: {save_path}")


if __name__ == "__main__":
    main()
