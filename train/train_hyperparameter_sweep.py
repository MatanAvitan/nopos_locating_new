"""
Hyperparameter Sweep Script
Tests stability across different hyperparameters (learning rate, batch size, initialization).
"""
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
sys.path.append('..')
from evaluation import evaluate
from transformer_lens import HookedTransformer, HookedTransformerConfig
from pathlib import Path
from utils import device
import os
from datetime import datetime
import numpy as np
import random
import json

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

N_CTX = 64
D_VOCAB = 5000
D_MODEL = 1024
D_MLP = 4096
EPOCHS = 150

# Hyperparameter grid (smart sampling, not full factorial)
HYPERPARAMS = [
    {"lr": 1e-3, "batch_size": 8192, "init_scale": 0.02},  # Baseline
    {"lr": 5e-4, "batch_size": 8192, "init_scale": 0.02},
    {"lr": 1e-3, "batch_size": 4096, "init_scale": 0.02},
    {"lr": 1e-3, "batch_size": 8192, "init_scale": 0.05},
    {"lr": 1e-3, "batch_size": 8192, "init_scale": 0.1},
]

def deactivate_position(model):
    model.pos_embed.W_pos.data[:] = 0.0
    model.pos_embed.W_pos.requires_grad = False

def freeze_embeddings(model):
    model.embed.W_E.requires_grad = False

def freeze_attention(model, l=0):
    for param in [model.blocks[l].attn.W_Q, model.blocks[l].attn.W_K, 
                  model.blocks[l].attn.W_V, model.blocks[l].attn.W_O,
                  model.blocks[l].attn.b_Q, model.blocks[l].attn.b_K,
                  model.blocks[l].attn.b_V, model.blocks[l].attn.b_O]:
        param.requires_grad = False

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
    def __init__(self, config, train_dataloader, val_dataloader, lr):
        super().__init__()
        self.model = HookedTransformer(config)
        self.model.to(device)
        deactivate_position(self.model)
        freeze_embeddings(self.model)
        freeze_attention(self.model, l=0)
        freeze_lns(self.model, l=0)
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self.lr = lr

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
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

def train_with_hyperparams(hparams):
    print(f"\n{'='*60}")
    print(f"Training: lr={hparams['lr']}, bs={hparams['batch_size']}, init={hparams['init_scale']}")
    print(f"{'='*60}\n")

    cfg = HookedTransformerConfig(
        n_layers=1, d_model=D_MODEL, d_head=D_MODEL, n_heads=1, d_mlp=D_MLP,
        d_vocab=D_VOCAB, n_ctx=N_CTX, act_fn='relu', normalization_type='LN',
        device=device, initializer_range=hparams['init_scale']
    )

    batch_size = hparams['batch_size']
    n_batches = max(50, 200000 // batch_size)
    
    train_tokens = torch.randint(0, D_VOCAB, (batch_size * n_batches, N_CTX))
    test_tokens = torch.randint(0, D_VOCAB, (batch_size * 20, N_CTX))
    labels_train = torch.arange(N_CTX).expand(train_tokens.size(0), -1)
    labels_test = torch.arange(N_CTX).expand(test_tokens.size(0), -1)

    train_dataset = TensorDataset(train_tokens, labels_train)
    test_dataset = TensorDataset(test_tokens, labels_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    lit_model = LitTransformer(cfg, train_loader, test_loader, hparams['lr'])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"hyperparam_lr{hparams['lr']}_bs{hparams['batch_size']}_init{hparams['init_scale']}"
    write_path = Path(f'models/hyperparameter_sweep/{name}_{timestamp}')
    write_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(dirpath=write_path, save_top_k=1, monitor='val_loss')
    trainer = Trainer(max_epochs=EPOCHS, accelerator='gpu', devices=1,
                     callbacks=[checkpoint_callback], accumulate_grad_batches=2, enable_progress_bar=False)

    start_time = datetime.now()
    trainer.fit(lit_model)
    training_time = (datetime.now() - start_time).total_seconds()

    results = evaluate(lit_model.model, test_loader, device)
    accuracy = float(results[0]) if isinstance(results, tuple) else float(results.get('accuracy', 0))

    results_dict = {
        'hyperparameters': hparams,
        'accuracy': accuracy,
        'training_time_seconds': training_time,
        'timestamp': timestamp
    }

    with open(write_path/'results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"✓ Accuracy: {accuracy:.4f}")
    del lit_model, trainer
    torch.cuda.empty_cache()
    return results_dict

def main():
    print("\n" + "="*60)
    print("HYPERPARAMETER SWEEP EXPERIMENT")
    print("="*60)
    
    all_results = []
    for hparams in HYPERPARAMS:
        try:
            result = train_with_hyperparams(hparams)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    output_file = Path('results/hyperparameter_sweep_results.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({'experiment': 'hyperparameter_sweep', 'results': all_results}, f, indent=2)

    print("\n" + "="*60)
    print("HYPERPARAMETER SUMMARY")
    print("="*60)
    for result in all_results:
        hp = result['hyperparameters']
        acc = result['accuracy']
        print(f"  lr={hp['lr']}, bs={hp['batch_size']}, init={hp['init_scale']}: acc={acc:.4f}")
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
