"""
Vocabulary Scaling Sweep Experiment
Tests vocabulary scaling from 1K to 32K to validate near-linear scaling claim.
Expected: min_samples ≈ 0.49 × vocab_size^0.98
"""

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
sys.path.append('..')
from evaluation import evaluate
from transformer_lens import HookedTransformer, HookedTransformerConfig
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from utils import device
import os
from datetime import datetime
import numpy as np
import random
import json

# Set random seeds for reproducibility
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

os.environ['CUDA_VISIBLE_DEVICES'] = '8'  # Use GPU 8 (longest running experiment)

BASE = Path('.').resolve()
N_CTX = 64
EPOCHS = 400
TARGET_ACCURACY = 0.95  # Early stopping threshold

# Vocabulary scaling parameters
VOCAB_SIZES = [1024, 2048, 4096, 8192, 16384, 32768]
SAMPLE_SIZES = [500, 1000, 2000, 5000, 10000, 20000, 50000]
D_MODEL_SCALE_FACTOR = 0.2  # d_model = vocab_size * scale_factor

def deactivate_position(model):
    model.pos_embed.W_pos.data[:] = 0.0
    model.pos_embed.W_pos.requires_grad = False

def freeze_embeddings(model):
    model.embed.W_E.requires_grad = False

def freeze_attention(model, l=0):
    model.blocks[l].attn.W_Q.requires_grad = False
    model.blocks[l].attn.W_K.requires_grad = False
    model.blocks[l].attn.W_V.requires_grad = False
    model.blocks[l].attn.W_O.requires_grad = False
    model.blocks[l].attn.b_Q.requires_grad = False
    model.blocks[l].attn.b_K.requires_grad = False
    model.blocks[l].attn.b_V.requires_grad = False
    model.blocks[l].attn.b_O.requires_grad = False

def freeze_lns(model, l=0):
    model.blocks[l].ln1.w.data[:] = 1
    model.blocks[l].ln1.w.requires_grad = False
    model.blocks[l].ln1.b.data[:] = 0
    model.blocks[l].ln1.b.requires_grad = False
    model.blocks[l].ln2.w.data[:] = 1
    model.blocks[l].ln2.w.requires_grad = False
    model.blocks[l].ln2.b.data[:] = 0
    model.blocks[l].ln2.b.requires_grad = False

class LitTransformer(pl.LightningModule):
    def __init__(self, config, train_dataloader, val_dataloader):
        super().__init__()
        self.model = HookedTransformer(config)
        self.model.to(device)
        deactivate_position(self.model)
        freeze_embeddings(self.model)
        freeze_attention(self.model, l=0)
        freeze_lns(self.model, l=0)
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self.best_val_acc = 0.0

    def forward(self, tokens):
        return self.model(tokens.to(device))

    def training_step(self, batch, batch_idx):
        tokens, targets = batch
        tokens = tokens.to(device)
        targets = targets.to(device)
        logits = self(tokens)
        loss = self.loss_fn(logits, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, targets = batch
        logits = self(tokens)
        loss = self.loss_fn(logits, targets)

        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean()

        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def loss_fn(self, logits, labels):
        log_probs = logits.log_softmax(-1)
        correct_log_probs = log_probs.gather(-1, labels[..., None])[..., 0]
        return -correct_log_probs.mean()


def run_experiment(vocab_size, n_samples):
    """
    Run a single vocab scaling experiment.

    Args:
        vocab_size: Vocabulary size
        n_samples: Number of training samples

    Returns:
        Dict with results
    """
    print(f"\n{'='*60}")
    print(f"Running: vocab_size={vocab_size}, n_samples={n_samples}")
    print(f"{'='*60}\n")

    # Scale model size with vocabulary
    d_model = int(vocab_size * D_MODEL_SCALE_FACTOR)
    d_mlp = d_model * 4
    batch_size = min(4096, n_samples // 10)  # Adaptive batch size

    # Create config
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=d_model,
        d_head=d_model,
        n_heads=1,
        d_mlp=d_mlp,
        d_vocab=vocab_size,
        n_ctx=N_CTX,
        act_fn='relu',
        normalization_type='LN',
        device=device
    )

    # Generate synthetic data
    train_tokens = torch.randint(low=0, high=vocab_size, size=(n_samples, N_CTX))
    test_tokens = torch.randint(low=0, high=vocab_size, size=(n_samples // 5, N_CTX))

    # Labels are positions
    train_labels = torch.arange(N_CTX).expand(train_tokens.size(0), -1)
    test_labels = torch.arange(N_CTX).expand(test_tokens.size(0), -1)

    train_dataset = TensorDataset(train_tokens, train_labels)
    test_dataset = TensorDataset(test_tokens, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    # Initialize model
    lit_model = LitTransformer(cfg, train_loader, test_loader)
    lit_model.to(device)

    # Setup trainer with early stopping
    experiment_name = f"vocab_scaling_v{vocab_size}_s{n_samples}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    write_path = Path(f'models/vocab_scaling_sweep/{experiment_name}_{timestamp}')
    write_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(write_path/'cfg', 'w') as f:
        f.write(str(cfg))

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=write_path,
        save_top_k=1,
        monitor='val_accuracy',
        mode='max'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.001,
        patience=20,
        mode='max',
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger('tblogs/', name=experiment_name)

    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        accumulate_grad_batches=2
    )

    # Train
    start_time = datetime.now()
    trainer.fit(lit_model)
    training_time = (datetime.now() - start_time).total_seconds()

    # Evaluate
    results = evaluate(lit_model.model, test_loader, device)
    accuracy = float(results[0]) if isinstance(results, tuple) else float(results.get('accuracy', 0))
    loss = float(results[1]) if isinstance(results, tuple) and len(results) > 1 else float(results.get('loss', 0))

    # Save results
    results_dict = {
        'vocab_size': vocab_size,
        'n_samples': n_samples,
        'd_model': d_model,
        'd_mlp': d_mlp,
        'accuracy': accuracy,
        'loss': loss,
        'training_time_seconds': training_time,
        'converged': accuracy >= TARGET_ACCURACY,
        'timestamp': timestamp
    }

    with open(write_path/'results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"✓ Completed: acc={accuracy:.4f}, loss={loss:.4f}, time={training_time:.1f}s")
    print(f"  Converged: {results_dict['converged']}")

    # Clean up to save memory
    del lit_model, trainer, train_loader, test_loader
    torch.cuda.empty_cache()

    return results_dict


def main():
    """
    Run vocab scaling sweep across all combinations.
    """
    all_results = []

    print("\n" + "="*60)
    print("VOCABULARY SCALING SWEEP EXPERIMENT")
    print("="*60)
    print(f"Vocab sizes: {VOCAB_SIZES}")
    print(f"Sample sizes: {SAMPLE_SIZES}")
    print(f"Target accuracy: {TARGET_ACCURACY}")
    print("="*60 + "\n")

    for vocab_size in VOCAB_SIZES:
        print(f"\n{'#'*60}")
        print(f"# VOCABULARY SIZE: {vocab_size}")
        print(f"{'#'*60}\n")

        # For each vocab size, find minimum samples needed
        min_samples_found = None

        for n_samples in SAMPLE_SIZES:
            try:
                result = run_experiment(vocab_size, n_samples)
                all_results.append(result)

                # Check if we've found minimum samples
                if result['converged'] and min_samples_found is None:
                    min_samples_found = n_samples
                    print(f"\n{'*'*60}")
                    print(f"* Found minimum samples for vocab {vocab_size}: {n_samples}")
                    print(f"{'*'*60}\n")
                    # Could break here to save time, but let's collect more data

            except Exception as e:
                print(f"ERROR in vocab={vocab_size}, samples={n_samples}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if min_samples_found is None:
            print(f"\nWARNING: Did not find convergence for vocab {vocab_size}")

    # Save aggregated results
    results_file = Path('results/vocab_scaling_results.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'vocab_scaling_sweep',
            'target_accuracy': TARGET_ACCURACY,
            'vocab_sizes': VOCAB_SIZES,
            'sample_sizes': SAMPLE_SIZES,
            'results': all_results,
            'completed_at': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*60}")
    print(f"Total experiments run: {len(all_results)}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}\n")

    # Print summary
    print("\nSummary by vocab size:")
    for vocab_size in VOCAB_SIZES:
        vocab_results = [r for r in all_results if r['vocab_size'] == vocab_size]
        converged = [r for r in vocab_results if r['converged']]
        if converged:
            min_samples = min(r['n_samples'] for r in converged)
            print(f"  Vocab {vocab_size:6d}: min_samples = {min_samples:6d} ({min_samples/vocab_size:.2f}x)")
        else:
            print(f"  Vocab {vocab_size:6d}: DID NOT CONVERGE")


if __name__ == "__main__":
    main()
