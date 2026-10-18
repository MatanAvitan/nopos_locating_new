"""
Training Script for 1-Layer Mechanism MLP Variant
"""

import os
import time
import math
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F

from model_1layer_mechanism import OneLayerMechanismModel, OneLayerMechanismConfig


@dataclass
class TrainConfig:
    n_embd: int = 768
    n_head: int = 12
    block_size: int = 128
    vocab_size: int = 50304
    norm_type: str = "layernorm"
    max_iters: int = 20000
    batch_size: int = 64
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    device: str = "cuda"
    dtype: str = "bfloat16"
    wandb_log: bool = True
    wandb_project: str = "nope-1layer-mechanism"
    out_dir: str = "out-1layer-mechanism"
    eval_interval: int = 500
    eval_iters: int = 100
    bos_token_id: int = 50256


def get_batch(data, config, device):
    tokens_needed = config.block_size - 1
    ix = torch.randint(len(data) - tokens_needed, (config.batch_size,))
    sequences = []
    for i in ix:
        i = i.item()
        after_bos = data[i : i + tokens_needed].astype(np.int64)
        seq = np.concatenate([[config.bos_token_id], after_bos])
        sequences.append(torch.from_numpy(seq))
    x = torch.stack(sequences)
    y = torch.arange(config.block_size).unsqueeze(0).expand(config.batch_size, -1)
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data, config, device):
    model.eval()
    losses = []
    for _ in range(config.eval_iters):
        x, y = get_batch(data, config, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--max_iters", type=int, default=20000)
    args = parser.parse_args()

    config = TrainConfig(wandb_log=args.wandb, max_iters=args.max_iters)

    # Load data
    data_dir = "data/openwebtext"
    train_data = np.memmap(
        os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    # Setup model
    m_config = OneLayerMechanismConfig(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        norm_type=config.norm_type,
    )
    model = OneLayerMechanismModel(m_config)
    model.freeze_all_except_mlp()
    model.to(config.device)

    optimizer = model.configure_optimizers(
        config.weight_decay,
        config.learning_rate,
        (config.beta1, config.beta2),
        config.device,
    )

    if config.wandb_log:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name="1layer_frozen_mlp_train",
            config=asdict(config),
        )

    best_val_loss = float("inf")
    os.makedirs(config.out_dir, exist_ok=True)

    for iter_num in range(config.max_iters + 1):
        x, y = get_batch(train_data, config, config.device)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_num % config.eval_interval == 0:
            val_loss = evaluate(model, val_data, config, config.device)
            print(
                f"iter {iter_num}: train_loss {loss.item():.4f}, val_loss {val_loss:.4f}"
            )
            if config.wandb_log:
                wandb.log(
                    {"iter": iter_num, "train_loss": loss.item(), "val_loss": val_loss}
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(), os.path.join(config.out_dir, "best_ckpt.pt")
                )

    if config.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    main()
