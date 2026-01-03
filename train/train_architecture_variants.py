"""
Architecture Variants Training Script
Tests robustness across different architectural choices (Complementary Experiment C2).
Varies number of heads and MLP sizes to test mechanism generalizability.
"""

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, TensorDataset
import torch
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

# Set random seeds
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# Configuration
N_CTX = 64
D_VOCAB = 5000
EPOCHS = 200
BATCH_SIZE = 8192
N_BATCHES = 100

# Architecture variants to test
ARCHITECTURES = [
    {"name": "baseline_1head", "n_heads": 1, "d_model": 1024, "d_mlp": 4096},
    {"name": "multihead_4heads", "n_heads": 4, "d_model": 1024, "d_mlp": 4096},
    {"name": "multihead_8heads", "n_heads": 8, "d_model": 1024, "d_mlp": 4096},
    {"name": "small_mlp", "n_heads": 1, "d_model": 1024, "d_mlp": 2048},
    {"name": "large_mlp", "n_heads": 1, "d_model": 1024, "d_mlp": 8192},
]


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

    def forward(self, tokens):
        return self.model(tokens.to(device))

    def training_step(self, batch, batch_idx):
        tokens, targets = batch
        logits = self(tokens.to(device))
        loss = self.loss_fn(logits, targets.to(device))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, targets = batch
        logits = self(tokens.to(device))
        loss = self.loss_fn(logits, targets.to(device))
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
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


def train_architecture_variant(arch_config):
    """Train model with specific architecture."""
    print(f"\n{'='*60}")
    print(f"Training: {arch_config['name']}")
    print(f"{'='*60}\n")

    d_model = arch_config['d_model']
    n_heads = arch_config['n_heads']
    d_head = d_model // n_heads

    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        d_mlp=arch_config['d_mlp'],
        d_vocab=D_VOCAB,
        n_ctx=N_CTX,
        act_fn='relu',
        normalization_type='LN',
        device=device
    )

    # Generate data
    train_tokens = torch.randint(0, D_VOCAB, (BATCH_SIZE * N_BATCHES, N_CTX))
    test_tokens = torch.randint(0, D_VOCAB, (BATCH_SIZE * N_BATCHES, N_CTX))
    labels = torch.arange(N_CTX).expand(train_tokens.size(0), -1)

    train_dataset = TensorDataset(train_tokens, labels)
    test_dataset = TensorDataset(test_tokens, labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)

    # Initialize model
    lit_model = LitTransformer(cfg, train_loader, test_loader)

    # Setup trainer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    write_path = Path(f'models/architecture_variants/{arch_config["name"]}_{timestamp}')
    write_path.mkdir(parents=True, exist_ok=True)

    with open(write_path/'cfg', 'w') as f:
        f.write(str(cfg))

    checkpoint_callback = ModelCheckpoint(dirpath=write_path, save_top_k=1, monitor='val_loss')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger('tblogs/', name=arch_config['name'])

    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
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
        'architecture': arch_config,
        'accuracy': accuracy,
        'loss': loss,
        'training_time_seconds': training_time,
        'timestamp': timestamp
    }

    with open(write_path/'results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"✓ {arch_config['name']}: accuracy={accuracy:.4f}")

    del lit_model, trainer
    torch.cuda.empty_cache()

    return results_dict


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("ARCHITECTURE VARIANTS EXPERIMENT")
    print("="*60)

    all_results = []

    for arch in ARCHITECTURES:
        try:
            result = train_architecture_variant(arch)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR with {arch['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Save aggregated results
    output_file = Path('results/architecture_variants_results.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'experiment': 'architecture_variants',
            'results': all_results
        }, f, indent=2)

    print("\n" + "="*60)
    print("ARCHITECTURE VARIANTS SUMMARY")
    print("="*60)
    for result in all_results:
        name = result['architecture']['name']
        acc = result['accuracy']
        print(f"  {name:20s}: accuracy={acc:.4f}")
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
