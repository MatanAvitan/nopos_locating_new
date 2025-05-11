#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from transformer_lens import HookedTransformer, HookedTransformerConfig, utils
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
import os

from scipy.linalg import norm
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils
import torch
import numpy as np
import plotly.express as px
import plotly.io as pio
from jaxtyping import Float, Int # Read about this library
import tqdm.auto as tqdm
import einops
from transformer_lens.utils import to_numpy
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
# import pytorch_lightning as pl
from functools import partial
import pandas as pd

from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities

# Utils
def line(tensor, line_labels=None, yaxis="", xaxis="", title="", legend_title="", **kwargs):
    tensor = to_numpy(tensor)
    # Use Plotly's white template and enforce a clean layout
    fig = px.line(tensor, template="plotly_white", **kwargs)
    
    # Update layout with enhanced aesthetics
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, family="Serif")),
        xaxis=dict(title=xaxis, title_font=dict(size=18), tickfont=dict(size=16)),
        yaxis=dict(title=yaxis, title_font=dict(size=18), tickfont=dict(size=16)),
        legend=dict(title=legend_title, font=dict(size=16)),
        width=800,
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Increase default line width and set marker styles
    for trace in fig.data:
        trace.line.width = 3
        trace.marker = dict(symbol="circle", size=8)
    
    # Apply line labels if provided
    if line_labels:
        for c, label in enumerate(line_labels):
            fig.data[c].name = label
    
    fig.show()


def imshow(tensor, yaxis="", xaxis="", **kwargs):
    tensor = to_numpy(tensor)
    # Use a high-quality continuous color scale and a white template
    plot_kwargs = {
        "color_continuous_scale": "RdBu_r",
        "color_continuous_midpoint": 0.0,
        "labels": {"x": xaxis, "y": yaxis},
        "template": "plotly_white",
        "aspect": "equal"
    }
    if kwargs['title']:
        title = kwargs['title']
    else:
        title = 'Image'
    plot_kwargs.update(kwargs)
    fig = px.imshow(tensor, **plot_kwargs)
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, family="Serif")),
        xaxis=dict(title=xaxis, title_font=dict(size=18), tickfont=dict(size=16)),
        yaxis=dict(title=yaxis, title_font=dict(size=18), tickfont=dict(size=16)),
        width=800,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    fig.show()
device = "cuda" if torch.cuda.is_available() else "cpu"
IS_FIRST = False


# # Load trained model

# # Uniformally sampled tokens

# In[ ]:


from tqdm import tqdm

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


class LitTransformer(pl.LightningModule):
    def __init__(self, config, train_dataloader, val_dataloader):
        super().__init__()
        self.model = HookedTransformer(config)
        self.model.to(device)
        deactivate_position(self.model)
        freeze_embeddings(self.model)
        freeze_attention(self.model, l=0)
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

    def forward(self, tokens):
        return self.model(tokens.to(device))

    def training_step(self, batch, batch_idx):
        tokens, targets = batch
        tokens = tokens.to(device)
        targets = targets .to(device)
        logits = self(tokens)
        loss = self.loss_fn(logits, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, targets = batch
        logits = self(tokens)
        loss = self.loss_fn(logits, targets)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def loss_fn(self, logits, labels, per_token=False):
        log_probs = logits.log_softmax(-1)
        correct_log_probs = log_probs.gather(-1, labels[..., None])[..., 0]
        if per_token:
            return -correct_log_probs
        else:
            return -correct_log_probs.mean()
################ Data
class CustomDataset(TensorDataset):
    def __init__(self, tokens, targets):
        super().__init__(tokens, targets)
BASE = Path('.').resolve() 
N_CTX = 64
TRAIN_RATIO = 0.8
TBLOGSDIR = f'tblogs'
D_VOCAB = 50_000
D_MODEL = 2_048
# N_SAMPLES = 1_024
N_SAMPLES = D_VOCAB*2
BATCH_SIZE = 64
EMB_STD = 0.025

train_dataset = torch.randint(low=0, high=D_VOCAB, size=(N_SAMPLES, N_CTX), device=device)
test_dataset = torch.randint(low=0, high=D_VOCAB, size=(N_SAMPLES, N_CTX), device=device)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

################ Data
cfg = HookedTransformerConfig(
    n_layers=1,
    d_model=D_MODEL,
    d_head=D_MODEL,
    n_heads=1,
    d_mlp=D_MODEL*4,
    d_vocab=D_VOCAB,
    n_ctx=N_CTX,
    act_fn='relu',
    normalization_type='LNPre',
    device=device,
    use_hook_mlp_in=True
)

def update_embedding(x):
    import torch.nn as nn
    return nn.init.normal_(x, mean=0.0, std=1/5_000)

lit_model = LitTransformer(cfg, train_loader, test_loader)
lit_model.to(device)
model = lit_model.model
# update_embedding(model.embed.W_E)
layers_to_cache = ['blocks.0.ln2.hook_normalized']
cached_act = {}
cached_act['post_ln2'] = []
for batch in tqdm(train_loader):
    _, train_cache = model.run_with_cache(batch, names_filter=layers_to_cache)
    act = train_cache['blocks.0.ln2.hook_normalized'].detach().cpu().numpy()
    cached_act['post_ln2'].append(act)
    del train_cache  # Free up memory
    torch.cuda.empty_cache()  # Clear GPU cache
cached_act['post_ln2'] = np.vstack(cached_act['post_ln2'])
# test_logits, test_cache = model.run_with_cache(test_loader.dataset[:100], names_filter=layers_to_cache)


# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.cuda.amp import GradScaler, autocast

# ——— Hyperparameters ———
TEST_AMOUNT_OF_SAMPLES = 1_024
BATCH_SIZE            = 64
EPOCHS                = 1_000
BASE_LR               = 1e-3
WEIGHT_DECAY          = 1e-2
WARMUP_FRAC           = 0.10  # first 10% of steps warm up

# ——— Model Definition ———
class PositionPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.mlp(x)

# ——— Setup ———
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim  = cached_act['post_ln2'].shape[-1]
hidden_dim = 4 * input_dim
output_dim = N_CTX

model     = PositionPredictorMLP(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
scaler    = GradScaler()

# ——— Prepare data ON CPU ———
# Note: we do NOT call `.to(device)` here
acts_cpu      = torch.tensor(cached_act['post_ln2'])            # [N_SAMPLES, N_CTX, D_MODEL] on CPU
num_train     = acts_cpu.size(0) - TEST_AMOUNT_OF_SAMPLES
train_acts, test_acts = torch.split(acts_cpu, [num_train, TEST_AMOUNT_OF_SAMPLES], dim=0)

class PositionDataset(Dataset):
    def __init__(self, activations: torch.Tensor):
        self.acts    = activations
        self.seq_len = activations.size(1)
    def __len__(self):
        return self.acts.size(0) * self.seq_len
    def __getitem__(self, idx: int):
        seq_idx, pos = divmod(idx, self.seq_len)
        x = self.acts[seq_idx, pos]   # still on CPU
        y = pos
        return x, y

train_ds     = PositionDataset(train_acts)
test_ds      = PositionDataset(test_acts)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    pin_memory=True, num_workers=4, prefetch_factor=2
)
test_loader  = DataLoader(
    test_ds, batch_size=BATCH_SIZE,
    pin_memory=True, num_workers=4, prefetch_factor=2
)

# ——— LR scheduling with linear warmup → linear decay ———
total_steps  = EPOCHS * len(train_loader)
warmup_steps = int(WARMUP_FRAC * total_steps)
def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return max(
        0.0,
        float(total_steps - step) /
        float(max(1, total_steps - warmup_steps))
    )
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ——— Training loop ———
global_step = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = correct = total = 0

    for Xb_cpu, yb_cpu in train_loader:
        # move each batch to GPU
        Xb = Xb_cpu.to(device, non_blocking=True)
        yb = yb_cpu.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            logits = model(Xb)
            loss   = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        global_step += 1

        running_loss += loss.item() * Xb.size(0)
        preds        = logits.argmax(dim=1)
        correct     += (preds == yb).sum().item()
        total       += yb.size(0)

    train_loss = running_loss / total
    train_acc  = correct      / total

    # — Validation —
    model.eval()
    val_loss = val_correct = val_total = 0

    with torch.no_grad():
        for Xv_cpu, yv_cpu in test_loader:
            Xv = Xv_cpu.to(device, non_blocking=True)
            yv = yv_cpu.to(device, non_blocking=True)
            with autocast():
                logits = model(Xv)
                loss   = criterion(logits, yv)

            val_loss    += loss.item() * Xv.size(0)
            preds        = logits.argmax(dim=1)
            val_correct += (preds == yv).sum().item()
            val_total   += yv.size(0)

    val_loss = val_loss / val_total
    val_acc  = val_correct / val_total
    current_lr = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch:03d}/{EPOCHS:03d}  "
        f"LR: {current_lr:.2e}  "
        f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
        f"Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}"
    )


# # Natural Language Tokens

# In[ ]:


from transformers import GPT2TokenizerFast
# ——— 1) Load & tokenize a real dataset ———
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # ensure there’s a pad token


# In[ ]:


# from tqdm import tqdm

# def deactivate_position(model):
#     model.pos_embed.W_pos.data[:] = 0.0
#     model.pos_embed.W_pos.requires_grad = False

# def freeze_embeddings(model):
#     model.embed.W_E.requires_grad = False

# def freeze_attention(model, l=0):
#     model.blocks[l].attn.W_Q.requires_grad = False
#     model.blocks[l].attn.W_K.requires_grad = False
#     model.blocks[l].attn.W_V.requires_grad = False
#     model.blocks[l].attn.W_O.requires_grad = False

#     model.blocks[l].attn.b_Q.requires_grad = False
#     model.blocks[l].attn.b_K.requires_grad = False
#     model.blocks[l].attn.b_V.requires_grad = False
#     model.blocks[l].attn.b_O.requires_grad = False


# class LitTransformer(pl.LightningModule):
#     def __init__(self, config, train_dataloader, val_dataloader):
#         super().__init__()
#         self.model = HookedTransformer(config)
#         self.model.to(device)
#         deactivate_position(self.model)
#         freeze_embeddings(self.model)
#         freeze_attention(self.model, l=0)
#         self._train_dataloader = train_dataloader
#         self._val_dataloader = val_dataloader

#     def forward(self, tokens):
#         return self.model(tokens.to(device))

#     def training_step(self, batch, batch_idx):
#         tokens, targets = batch
#         tokens = tokens.to(device)
#         targets = targets .to(device)
#         logits = self(tokens)
#         loss = self.loss_fn(logits, targets)
#         self.log('train_loss', loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         tokens, targets = batch
#         logits = self(tokens)
#         loss = self.loss_fn(logits, targets)
#         self.log('val_loss', loss)

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1)
#         scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
#         return [optimizer], [scheduler]

#     def train_dataloader(self):
#         return self._train_dataloader

#     def val_dataloader(self):
#         return self._val_dataloader

#     def loss_fn(self, logits, labels, per_token=False):
#         log_probs = logits.log_softmax(-1)
#         correct_log_probs = log_probs.gather(-1, labels[..., None])[..., 0]
#         if per_token:
#             return -correct_log_probs
#         else:
#             return -correct_log_probs.mean()
# ################ Data
# class CustomDataset(TensorDataset):
#     def __init__(self, tokens, targets):
#         super().__init__(tokens, targets)
# BASE = Path('.').resolve() 
# N_CTX = 64
# TBLOGSDIR = f'tblogs'
# D_VOCAB = tokenizer.vocab_size 
# D_MODEL = 2_048
# # N_SAMPLES = 1_024
# TEST_AMOUNT_OF_SAMPLES = 1_024
# N_SAMPLES = D_VOCAB + TEST_AMOUNT_OF_SAMPLES # Later I will uset the last 1024 samples for testing
# BATCH_SIZE = 64
# # EMB_STD = 0.025

# train_dataset = torch.randint(low=0, high=D_VOCAB, size=(N_SAMPLES, N_CTX), device=device)
# test_dataset = torch.randint(low=0, high=D_VOCAB, size=(N_SAMPLES, N_CTX), device=device)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ################ Data
# cfg = HookedTransformerConfig(
#     n_layers=1,
#     d_model=D_MODEL,
#     d_head=D_MODEL,
#     n_heads=1,
#     d_mlp=D_MODEL*4,
#     d_vocab=tokenizer.vocab_size,
#     n_ctx=N_CTX,
#     act_fn='relu',
#     normalization_type='LNPre',
#     device=device,
#     use_hook_mlp_in=True
# )

# def update_embedding(x):
#     import torch.nn as nn
#     return nn.init.normal_(x, mean=0.0, std=1/5_000)

# lit_model = LitTransformer(cfg, train_loader, test_loader)
# lit_model.to(device)
# model = lit_model.model
# # update_embedding(model.embed.W_E)
# layers_to_cache = ['blocks.0.ln2.hook_normalized']
# cached_act = {}
# cached_act['post_ln2'] = []
# for batch in tqdm(train_loader):
#     _, train_cache = model.run_with_cache(batch, names_filter=layers_to_cache)
#     act = train_cache['blocks.0.ln2.hook_normalized'].detach().cpu().numpy()
#     cached_act['post_ln2'].append(act)
#     del train_cache  # Free up memory
#     torch.cuda.empty_cache()  # Clear GPU cache
# cached_act['post_ln2'] = np.vstack(cached_act['post_ln2'])


# ## Initialize the model

# In[ ]:


from pathlib import Path
BASE = Path('.').resolve() 
N_CTX = 64
TBLOGSDIR = f'tblogs'
D_VOCAB = tokenizer.vocab_size 
D_MODEL = 2_048
TEST_AMOUNT_OF_SAMPLES = 1_024
N_SAMPLES = D_VOCAB + TEST_AMOUNT_OF_SAMPLES # Later I will uset the last 1024 samples for testing
BATCH_SIZE = 64


# In[ ]:


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


class LitTransformer(pl.LightningModule):
    def __init__(self, config, train_dataloader, val_dataloader):
        super().__init__()
        self.model = HookedTransformer(config)
        self.model.to(device)
        deactivate_position(self.model)
        freeze_embeddings(self.model)
        freeze_attention(self.model, l=0)
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

    def forward(self, tokens):
        return self.model(tokens.to(device))

    def training_step(self, batch, batch_idx):
        tokens, targets = batch
        tokens = tokens.to(device)
        targets = targets .to(device)
        logits = self(tokens)
        loss = self.loss_fn(logits, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, targets = batch
        logits = self(tokens)
        loss = self.loss_fn(logits, targets)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def loss_fn(self, logits, labels, per_token=False):
        log_probs = logits.log_softmax(-1)
        correct_log_probs = log_probs.gather(-1, labels[..., None])[..., 0]
        if per_token:
            return -correct_log_probs
        else:
            return -correct_log_probs.mean()
cfg = HookedTransformerConfig(
    n_layers=1,
    d_model=D_MODEL,
    d_head=D_MODEL,
    n_heads=1,
    d_mlp=D_MODEL*4,
    d_vocab=D_VOCAB,
    n_ctx=N_CTX,
    act_fn='relu',
    normalization_type='LNPre',
    device=device,
    use_hook_mlp_in=True
)


# In[ ]:


import os
import glob
import torch
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

TEST_AMOUNT_OF_SAMPLES = 1_024
# ——— Hyperparams & Constants ———
N_CTX                 = 64
BATCH_SIZE            = 64
EPOCHS                = 1_000
BASE_LR               = 1e-3
WEIGHT_DECAY          = 1e-2
WARMUP_FRAC           = 0.10  # first 10% steps warm up

cache_dir = '/dccstor/ai_security2/matan/cache_dir'

# Load Wikitext-2 (you can swap for any HF text dataset)
# raw = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
raw = load_dataset('ccdv/arxiv-summarization', split='train', cache_dir=cache_dir)
# raw = raw[:N_SAMPLES] 

def tokenize_fn(examples):
    return tokenizer(
        examples['abstract'],
        truncation=True,
        padding='max_length',
        max_length=N_CTX
    )

# Tokenize in batches and drop the original text
tok = raw.map(tokenize_fn, batched=True, remove_columns=['abstract'])
input_ids = torch.tensor(tok['input_ids'], dtype=torch.long)  # [num_docs, N_CTX]

# ——— 2) Split into train / test by **whole** examples ———
train_ids    = input_ids[:-TEST_AMOUNT_OF_SAMPLES]
test_ids     = input_ids[-TEST_AMOUNT_OF_SAMPLES:]

# ——— 3) Build TensorDatasets for LM training (tokens → targets) ———
# Here we predict each token from itself (i.e. next‐token LM shifted inside your LitModule)
train_ds = TensorDataset(train_ids, train_ids)
test_ds  = TensorDataset(test_ids,  test_ids)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    pin_memory=True, num_workers=16, prefetch_factor=2
)
test_loader  = DataLoader(
    test_ds,  batch_size=BATCH_SIZE,
    pin_memory=True, num_workers=16, prefetch_factor=2
)

# ——— 4) Rest of your setup (model / optim / training) stays exactly the same ———
lit_model = LitTransformer(cfg, train_loader, test_loader)
lit_model.to(device)
model = lit_model.model
# update_embedding(model.embed.W_E)
layers_to_cache = ['blocks.0.ln2.hook_normalized']
# cache_acts_dir = "/dccstor/ai_security2/matan/cached_activations"
# os.makedirs(cache_acts_dir, exist_ok=True)


# ## Save the activations

# In[ ]:


from torch.utils.data import Dataset, DataLoader
def store_acts(loader, cache_acts_dir):
    model.eval()
    with torch.no_grad():
        for i, (tokens, _) in enumerate(tqdm(loader, desc="Caching loader")):
            tokens = tokens.to(device, non_blocking=True)
            try:
                _, cache = model.run_with_cache(tokens, names_filter=layers_to_cache)
                act = cache['blocks.0.ln2.hook_normalized'].detach().cpu()  # [B, N_CTX, D_MODEL]

                # Save to disk batch-wise
                torch.save(act, f"{cache_acts_dir}/batch_{i:05d}.pt")

                del cache, act
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"Skipping batch {i} due to OOM.")
                    torch.cuda.empty_cache()
                else:
                    raise

def load_acts_loader(cache_acts_dir):
    class CachedActivationDataset(Dataset):
        def __init__(self, file_paths):
            self.file_paths = file_paths
            self.index_map = []
            for file_idx, path in enumerate(file_paths):
                acts = torch.load(path, map_location='cpu')
                B, N, _ = acts.shape
                for i in range(B * N):
                    self.index_map.append((file_idx, i))
            self.seq_len = N

        def __len__(self):
            return len(self.index_map)

        def __getitem__(self, idx):
            file_idx, flat_idx = self.index_map[idx]
            act = torch.load(self.file_paths[file_idx], map_location='cpu')  # [B, N, D]
            B, N, D = act.shape
            b_idx = flat_idx // N
            pos = flat_idx % N
            return act[b_idx, pos], pos  # [D_MODEL], int

    # ——— Reload cached activations for training ———
    acts_files = sorted(glob.glob(f"{cache_acts_dir}/batch_*.pt"))
    cached_ds = CachedActivationDataset(acts_files)
    cached_loader = DataLoader(
        cached_ds, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=True, num_workers=16, prefetch_factor=2
    )
    return cached_loader


# In[ ]:


train_cache_acts_dir = "/dccstor/ai_security2/matan/train_cached_activations"
os.makedirs(train_cache_acts_dir, exist_ok=True)
if IS_FIRST:
    store_acts(train_loader, train_cache_acts_dir)
train_cached_loader = load_acts_loader(train_cache_acts_dir)


# In[ ]:


test_cache_acts_dir = "/dccstor/ai_security2/matan/test_cached_activations"
os.makedirs(test_cache_acts_dir, exist_ok=True)
if IS_FIRST:
    store_acts(test_loader, test_cache_acts_dir)
test_cached_loader = load_acts_loader(test_cache_acts_dir)


# In[ ]:


# model.eval()
# with torch.no_grad():
#     for i, (tokens, _) in enumerate(tqdm(train_loader, desc="Caching train")):
#         tokens = tokens.to(device, non_blocking=True)
#         try:
#             _, cache = model.run_with_cache(tokens, names_filter=layers_to_cache)
#             act = cache['blocks.0.ln2.hook_normalized'].detach().cpu()  # [B, N_CTX, D_MODEL]

#             # Save to disk batch-wise
#             torch.save(act, f"{cache_acts_dir}/batch_{i:05d}.pt")

#             del cache, act
#             torch.cuda.empty_cache()

#         except RuntimeError as e:
#             if 'CUDA out of memory' in str(e):
#                 print(f"Skipping batch {i} due to OOM.")
#                 torch.cuda.empty_cache()
#             else:
#                 raise
# # files = sorted(glob.glob(f"{cache_acts_dir}/batch_*.pt"))
# # acts_list = [torch.load(f) for f in files]
# # cached_act = {'post_ln2': torch.cat(acts_list, dim=0)}  # [total_batches * B, N_CTX, D_MODEL]


# In[ ]:


# from torch.utils.data import Dataset, DataLoader
# class CachedActivationDataset(Dataset):
#     def __init__(self, file_paths):
#         self.file_paths = file_paths
#         self.index_map = []
#         for file_idx, path in enumerate(file_paths):
#             acts = torch.load(path, map_location='cpu')
#             B, N, _ = acts.shape
#             for i in range(B * N):
#                 self.index_map.append((file_idx, i))
#         self.seq_len = N

#     def __len__(self):
#         return len(self.index_map)

#     def __getitem__(self, idx):
#         file_idx, flat_idx = self.index_map[idx]
#         act = torch.load(self.file_paths[file_idx], map_location='cpu')  # [B, N, D]
#         B, N, D = act.shape
#         b_idx = flat_idx // N
#         pos = flat_idx % N
#         return act[b_idx, pos], pos  # [D_MODEL], int

# # ——— Reload cached activations for training ———
# train_files = sorted(glob.glob(f"{cache_acts_dir}/batch_*.pt"))
# cached_train_ds = CachedActivationDataset(train_files)
# cached_train_loader = DataLoader(
#     cached_train_ds, batch_size=BATCH_SIZE, shuffle=True,
#     pin_memory=True, num_workers=4, prefetch_factor=2
# )


# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

TEST_AMOUNT_OF_SAMPLES = 1_024

# Define the MLP model
class PositionPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PositionPredictorMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# Set dimensions
# input_dim = next(iter(test_cached_loader))[0].shape[-1]
input_dim = 2_048
hidden_dim = 4 * input_dim
output_dim = N_CTX

# Move everything to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = PositionPredictorMLP(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=1e-3)

epochs     = 1_000
for epoch in tqdm(range(1, epochs+1)):
    # ——— Training ———
    mlp_model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for batch_X, batch_y in tqdm(train_cached_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = mlp_model(batch_X)
        loss    = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)
        preds        = outputs.argmax(dim=1)
        correct     += (preds == batch_y).sum().item()
        total       += batch_y.size(0)

    train_loss = running_loss / total
    train_acc  = correct      / total

    # ——— Validation ———
    mlp_model.eval()
    val_loss    = 0.0
    val_correct = 0
    val_total   = 0

    with torch.no_grad():
        for val_X, val_y in test_cached_loader:
            val_X, val_y = val_X.to(device), val_y.to(device)
            logits       = mlp_model(val_X)
            loss         = criterion(logits, val_y)

            val_loss    += loss.item() * val_X.size(0)
            preds        = logits.argmax(dim=1)
            val_correct += (preds == val_y).sum().item()
            val_total   += val_y.size(0)

    val_loss = val_loss / val_total
    val_acc  = val_correct / val_total

    print(f"Epoch {epoch}/{epochs}  "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
          f"Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


model.W_E.data.std()


# In[ ]:


cached_act['post_ln2'].shape


# In[ ]:


key = (model.W_E.data @ (model.W_V.data.squeeze(0).squeeze(0) @ model.W_O.data.squeeze(0).squeeze(0))).sum(axis=0).detach().cpu().numpy()
key.shape


# In[ ]:


cached_act['post_ln2'].shape


# In[ ]:


model.blocks[0].mlp.W_in


# In[ ]:


line(F.relu(torch.tensor(cached_act['post_ln2'][7], device='cuda')@model.blocks[0].mlp.W_in.data).detach().cpu().numpy() @ (key@model.blocks[0].mlp.W_in.data.detach().cpu().numpy()))


# In[ ]:


line((cached_act['post_ln2'][8]) @ key)


# In[ ]:


line(F.relu((train_cache['blocks.0.ln2.hook_normalized'] @ key)[22, :]))


# In[ ]:


line(F.relu((train_cache['blocks.0.ln2.hook_normalized'] @ model.W_E.data.T)[22, :]))


# In[ ]:


line(F.relu((train_cache['blocks.0.ln2.hook_normalized'] @ model.W_E.data.T)[22, :])>0)


# In[ ]:


line(F.relu((train_cache['blocks.0.ln2.hook_normalized'] @ model.W_E.data.T)[22, :]))


# In[ ]:


model.W_E.data @ model.W_V.data.squeeze(0).squeeze(0)


# In[ ]:


train_cache['blocks.0.ln2.hook_normalized'].shape


# In[ ]:


line(train_cache['blocks.0.ln2.hook_normalized'][0] @ (model.W_E.data @ model.W_V.data.squeeze(0).squeeze(0).T).T)


# In[ ]:


train_cache['blocks.0.ln2.hook_normalized'].shape, model.W_E.data.shape


# In[ ]:


train_loader.dataset[0, 0]


# In[ ]:


line(train_cache['blocks.0.ln2.hook_normalized'][0, 0] @ (model.W_E.data).T)


# In[ ]:


line((train_cache['blocks.0.ln2.hook_normalized'][0, 2, :] @ model.W_E.data.T))


# In[ ]:


line((train_cache['blocks.0.ln2.hook_normalized'][0, 2, :] @ model.W_E.data.T))


# In[ ]:


train_loader.dataset[0,23]


# In[ ]:


std


# In[ ]:


model.W_E.data


# In[ ]:


line(train_cache['blocks.0.ln2.hook_normalized'][sample, pos, :] @ ((model.W_E.data).T))


# In[ ]:


model.W_E.data.std()


# In[ ]:


sample = 10
pos = 2
train_cache['blocks.0.ln2.hook_normalized'][sample, pos, :]


# In[ ]:


line(train_cache['blocks.0.ln2.hook_normalized'].mean(0))


# In[ ]:


sample = 10
pos = 2
line(train_cache['blocks.0.ln2.hook_normalized'][sample, pos, :] @ ((model.W_E.data @ model.W_V.data.squeeze(0).squeeze(0)).T))


# In[ ]:


model.W_E.data.sum(axis=0).shape


# In[ ]:


sample = 23
pos = 16
line(train_cache['blocks.0.ln2.hook_normalized'][sample, range(64), :] @ ((model.W_E.data.sum(axis=0).unsqueeze(0) @ model.W_V.data.squeeze(0).squeeze(0)).T))


# In[ ]:


sample = 10
pos = 19

(train_cache['blocks.0.ln2.hook_normalized'][sample, pos, :] @ ((model.W_E.data.sum(axis=0).unsqueeze(0)).T))


# In[ ]:


train_loader.dataset[sample, :pos+1]


# In[ ]:


line(train_cache['blocks.0.ln2.hook_normalized'][0, 22] @ (model.W_E.data @ model.W_V.data.squeeze(0).squeeze(0)).T)


# In[ ]:


line((train_cache['blocks.0.ln2.hook_normalized'][0] @ (model.W_E.data @ model.W_V.data.squeeze(0).squeeze(0))))


# In[ ]:


line((train_cache['blocks.0.ln2.hook_normalized'][0] @ (model.W_E.data @ model.W_V.data.squeeze(0).squeeze(0))))


# In[ ]:


line(F.relu((train_cache['blocks.0.ln2.hook_normalized'] @ model.W_E.data.T)[3, :]))


# In[ ]:


(F.relu((train_cache['blocks.0.ln2.hook_normalized'] @ model.W_E.data.T)[22, :])>0)[:,2].sum()


# In[ ]:


(F.relu((train_cache['blocks.0.ln2.hook_normalized'] @ model.W_E.data.T)[22, :])>0)[0].sum()


# In[ ]:





# In[ ]:


sum(F.relu(((model.W_E.data) @ model.W_E.data.T)[22, :])>0)


# In[ ]:


sum(F.relu(((model.W_E.data @ model.W_V.data.squeeze(0).squeeze(0)) @ model.W_E.data.T)[22, :])>0)


# In[ ]:


line((((model.W_E.data @ model.W_V.data.squeeze(0).squeeze(0)) @ model.W_E.data.T)[22, :]))


# In[ ]:


std=model.W_E.data.std()
line((((model.W_E.data @ model.W_V.data.squeeze(0).squeeze(0)) @ model.W_E.data.T)[22, :])**2/1024)


# In[ ]:


std=model.W_E.data.std()
line((train_cache['blocks.0.ln2.hook_normalized'][0] @ model.W_E.data.T)[22, :]/std)


# In[ ]:


train_loader.dataset[0,22] # Boom!


# In[ ]:


line((train_cache['blocks.0.ln2.hook_normalized'][0] @ model.W_E.data.T)[23, :])


# In[ ]:


train_loader.dataset[0,23] # Boom!


# In[ ]:





# In[ ]:


model.W_E.data[986]


# In[ ]:


line((train_cache['blocks.0.ln2.hook_normalized'][0] @ model.W_E.data.T)[22, :])


# In[ ]:





# In[ ]:


line((train_cache['blocks.0.ln2.hook_normalized'][0] @ model.W_E.data.T).sum(axis=1))


# In[ ]:


line((train_cache['blocks.0.ln2.hook_normalized'][0] * model.W_E.data.T).mean(axis=1))


# # D_VOCAB = 1_024

# In[ ]:


model.embed.W_E.data.std(), model.blocks[0].attn.W_Q.data.std()


# In[ ]:


# update_embedding(model.embed.W_E)


# In[ ]:


model.embed.W_E.data.std(), model.blocks[0].attn.W_Q.data.std()


# In[ ]:


# L0
n_layers = 1
for layer_id in range(n_layers):
    # Heads
    for i in range(cfg.n_heads):
        imshow(
            (np.exp(train_cache[f"blocks.{layer_id}.attn.hook_attn_scores"][:, i, :, :].mean([0]).cpu().numpy())),
            title=f"Layer {layer_id} Attention Scores",
            height=500,
            width=500,
        )


# In[ ]:


# L0
n_layers = 1
for layer_id in range(n_layers):
    # Heads
    for i in range(cfg.n_heads):
        imshow(
            to_numpy(train_cache["attn", layer_id][:, [i], :, :].mean([0, 1])),
            title=f"Layer {layer_id} Attention Pattern {i}",
            height=500,
            width=500,
        )


# In[ ]:


line(train_cache['blocks.0.ln2.hook_normalized'].mean(0)[:,:100], title='Post Attn Layer Norm Output', legend_title='Neurons')


# In[ ]:


neurons_avg_across_pos = train_cache['blocks.0.ln2.hook_normalized'].mean(0) # 64 x 2048
# Let's find 10 neurons that are the closest to each other and also positive
neurons_mean = neurons_avg_across_pos.mean(0)
q = np.quantile(neurons_mean, 0.75)
top_quarter_neurons_ids = np.where(neurons_mean > q)[0]
line(neurons_avg_across_pos[:, top_quarter_neurons_ids], title='Top 25% neurons', xaxis='Position', yaxis='Neuron value', legend_title='Neurons', width=800, height=500)


# In[ ]:


print(regressors[0].coefficients)


# In[ ]:





# In[ ]:


avg_line_for_upper_quarter = neurons_avg_across_pos[:, top_quarter_neurons_ids].mean(1)
line(avg_line_for_upper_quarter, title='Average line for upper quarter neurons', xaxis='Position', yaxis='Activation', color_discrete_sequence=['blue'], legend_title=None)


# In[ ]:





# In[ ]:





# # Use top 25% of positive neurons as a label

# ## Use only 11 Neurons

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Y
def get_y(test_cache, q=0.9, layer='blocks.0.ln2.hook_normalized', draw_line=True):
    neurons_avg_across_pos = train_cache[layer].mean(0).to(device)  # Move to GPU
    neurons_mean = neurons_avg_across_pos.mean(0)
    q = np.quantile(neurons_mean.cpu().numpy(), q)  # Move to CPU for numpy operations
    top_quarter_neurons_ids = np.where(neurons_mean.cpu().numpy() > q)[0]
    
    # Print the number of neurons used to create Y
    print(f"Number of neurons used to create Y: {len(top_quarter_neurons_ids)}")
    
    if draw_line:
        line(neurons_avg_across_pos[:, top_quarter_neurons_ids].cpu().numpy())  # Move to CPU for plotting
    avg_line_for_upper_quarter = neurons_avg_across_pos[:, top_quarter_neurons_ids].mean(1)
    return avg_line_for_upper_quarter.cpu().numpy(), len(top_quarter_neurons_ids)  # Return Y and number of neurons

# Build X
def regress_samples_to_mean(train_cache, test_cache, q, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="./plots_neurons"):    
    # Get activations with shape: (batch, ctx, neurons)
    activations = train_cache[act].to(device)  # Move to GPU
    activations = activations.cpu().numpy()  # Move to CPU for sklearn compatibility

    num_samples, n_positions, n_neurons = activations.shape
    print("Activations shape:", activations.shape)

    # Get selected neurons ids
    neurons_avg_across_pos = activations.mean(0)
    neurons_mean = neurons_avg_across_pos.mean(0)
    q = np.quantile(neurons_mean, q)
    top_neurons_ids = np.where(neurons_mean > q)[0]

    # Flatten (batch, ctx) to get X, and compute thresholds using the given quantiles.
    # Create output directory if saving plots
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Train a linear regressor for each sample
    Y_sample = mean_curve.reshape(-1,1)  # Target line [64, 1]
    regressor = LinearRegression()

    print(f'N neurons used for training: {len(top_neurons_ids)}')
    n_neurons = len(top_neurons_ids)

    selected_activations = activations[:, :, top_neurons_ids]
    Y = np.tile(Y_sample, (num_samples, 1)).reshape(-1, 1)  # Shape: (num_samples * n_positions, 1)
    regressor.fit(selected_activations.reshape(-1, n_neurons), Y)
    # Predict the positions using the trained model
    for i in [0]:
        # Extract activations for the selected neurons
        selected_activations = test_cache[act][i][:, top_neurons_ids].to(device)  # Move to GPU
        predicted_positions = regressor.predict(selected_activations.cpu().numpy())  # Move to CPU for prediction
        # Plot the true positions vs. predicted positions for the first sample
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_positions), Y_sample, label=f'True Line (Neurons: {num_neurons})', color='blue', linestyle='--')
        plt.plot(range(n_positions), predicted_positions, label='Predicted Line', color='red')
        plt.xlabel('Position')
        plt.ylabel('Predicted Position')
        plt.title(f'Linear Regression Prediction of Positions for sample {i}\n')
        plt.legend()

        if save_to_disk:
            plot_path = os.path.join(output_dir, f"linear_regression_q={q}_plot.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        else:
            plt.show()

        plt.close()
    return regressor

regressors = []
# for q in [0.99,0.8,0.6,0.5, 0]:
for q in [0]:
    mean_curve, num_neurons = get_y(train_cache, q=0.75, layer='blocks.0.ln2.hook_normalized', draw_line=False)
    regressor = regress_samples_to_mean(train_cache, test_cache, q, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="activation_plots")
    regressors.append(regressor)


# In[ ]:


line(regressors[-1].coef_.T)


# In[ ]:


regressors[-1].coef_.shape


# In[ ]:


q95=np.quantile(regressors[-1].coef_, 0.95)
q5=np.quantile(regressors[-1].coef_, 0.05)
np.where(regressors[-1].coef_ > q95)[0]
np.where(regressors[-1].coef_ < q5)[0]
union_neurons = np.where((regressors[-1].coef_ > q95) | (regressors[-1].coef_ < q5))[0]
union_neurons.shape


# # Train the LR on the entire neurons to get the regressor coefficients

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Y
def get_y(test_cache, q=0.9, layer='blocks.0.ln2.hook_normalized', draw_line=True):
    neurons_avg_across_pos = train_cache[layer].mean(0).to(device)  # Move to GPU
    neurons_mean = neurons_avg_across_pos.mean(0)
    q = np.quantile(neurons_mean.cpu().numpy(), q)  # Move to CPU for numpy operations
    top_quarter_neurons_ids = np.where(neurons_mean.cpu().numpy() > q)[0]
    
    # Print the number of neurons used to create Y
    print(f"Number of neurons used to create Y: {len(top_quarter_neurons_ids)}")
    
    if draw_line:
        line(neurons_avg_across_pos[:, top_quarter_neurons_ids].cpu().numpy())  # Move to CPU for plotting
    avg_line_for_upper_quarter = neurons_avg_across_pos[:, top_quarter_neurons_ids].mean(1)
    return avg_line_for_upper_quarter.cpu().numpy(), len(top_quarter_neurons_ids)  # Return Y and number of neurons

# Build X
def regress_samples_to_mean(train_cache, test_cache, q, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="./plots_neurons"):    
    # Get activations with shape: (batch, ctx, neurons)
    activations = train_cache[act].to(device)  # Move to GPU
    activations = activations.cpu().numpy()  # Move to CPU for sklearn compatibility

    num_samples, n_positions, n_neurons = activations.shape
    print("Activations shape:", activations.shape)


    # Flatten (batch, ctx) to get X, and compute thresholds using the given quantiles.
    # Create output directory if saving plots
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Train a linear regressor for each sample
    Y_sample = mean_curve.reshape(-1,1)  # Target line [64, 1]
    regressor = LinearRegression()

    top_neurons_ids = range(n_neurons)
    print(f'N neurons used for training: {len(top_neurons_ids)}')
    n_neurons = len(top_neurons_ids)

    selected_activations = activations[:, :, top_neurons_ids]
    Y = np.tile(Y_sample, (num_samples, 1)).reshape(-1, 1)  # Shape: (num_samples * n_positions, 1)
    regressor.fit(selected_activations.reshape(-1, n_neurons), Y)
    # Predict the positions using the trained model
    for i in [0]:
        # Extract activations for the selected neurons
        selected_activations = test_cache[act][i][:, top_neurons_ids].to(device)  # Move to GPU
        predicted_positions = regressor.predict(selected_activations.cpu().numpy())  # Move to CPU for prediction
        # Plot the true positions vs. predicted positions for the first sample
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_positions), Y_sample, label=f'True Line (Neurons: {num_neurons})', color='blue', linestyle='--')
        plt.plot(range(n_positions), predicted_positions, label='Predicted Line', color='red')
        plt.xlabel('Position')
        plt.ylabel('Predicted Position')
        plt.title(f'Linear Regression Prediction of Positions for sample {i}\n')
        plt.legend()

        if save_to_disk:
            plot_path = os.path.join(output_dir, f"linear_regression_q={q}_plot.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        else:
            plt.show()

        plt.close()
    return regressor

regressors = []
# for q in [0.99,0.8,0.6,0.5, 0]:
for q in [0]:
    mean_curve, num_neurons = get_y(train_cache, q=0.75, layer='blocks.0.ln2.hook_normalized', draw_line=False)
    regressor = regress_samples_to_mean(train_cache, test_cache, q, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="activation_plots")
    regressors.append(regressor)


# In[ ]:


best_regressor = regressors[-1]


# ## Use the strongest neurons in terms of LR coefficients

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Y
def get_y(test_cache, q=0.9, layer='blocks.0.ln2.hook_normalized', draw_line=True):
    neurons_avg_across_pos = train_cache[layer].mean(0).to(device)  # Move to GPU
    neurons_mean = neurons_avg_across_pos.mean(0)
    q = np.quantile(neurons_mean.cpu().numpy(), q)  # Move to CPU for numpy operations
    top_quarter_neurons_ids = np.where(neurons_mean.cpu().numpy() > q)[0]
    
    # Print the number of neurons used to create Y
    print(f"Number of neurons used to create Y: {len(top_quarter_neurons_ids)}")
    
    if draw_line:
        line(neurons_avg_across_pos[:, top_quarter_neurons_ids].cpu().numpy())  # Move to CPU for plotting
    avg_line_for_upper_quarter = neurons_avg_across_pos[:, top_quarter_neurons_ids].mean(1)
    return avg_line_for_upper_quarter.cpu().numpy(), len(top_quarter_neurons_ids)  # Return Y and number of neurons

# Build X
def regress_samples_to_mean(train_cache, test_cache, q, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="./plots_neurons"):    
    # Get activations with shape: (batch, ctx, neurons)
    activations = train_cache[act].to(device)  # Move to GPU
    activations = activations.cpu().numpy()  # Move to CPU for sklearn compatibility

    num_samples, n_positions, n_neurons = activations.shape
    print("Activations shape:", activations.shape)

    # Get selected neurons ids
    q95=np.quantile(best_regressor.coef_, 0.75)
    q5=np.quantile(best_regressor.coef_, 0.25)
    np.where(best_regressor.coef_ > q95)[0]
    np.where(best_regressor.coef_ < q5)[0]
    top_neurons_ids = np.where((best_regressor.coef_ > q95) | (best_regressor.coef_ < q5))[0]

    # Flatten (batch, ctx) to get X, and compute thresholds using the given quantiles.
    # Create output directory if saving plots
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Train a linear regressor for each sample
    Y_sample = mean_curve.reshape(-1,1)  # Target line [64, 1]
    regressor = LinearRegression()

    print(f'N neurons used for training: {len(top_neurons_ids)}')
    n_neurons = len(top_neurons_ids)

    selected_activations = activations[:, :, top_neurons_ids]
    Y = np.tile(Y_sample, (num_samples, 1)).reshape(-1, 1)  # Shape: (num_samples * n_positions, 1)
    regressor.fit(selected_activations.reshape(-1, n_neurons), Y)
    # Predict the positions using the trained model
    for i in [0]:
        # Extract activations for the selected neurons
        selected_activations = test_cache[act][i][:, top_neurons_ids].to(device)  # Move to GPU
        predicted_positions = regressor.predict(selected_activations.cpu().numpy())  # Move to CPU for prediction
        # Plot the true positions vs. predicted positions for the first sample
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_positions), Y_sample, label=f'True Line (Neurons: {num_neurons})', color='blue', linestyle='--')
        plt.plot(range(n_positions), predicted_positions, label='Predicted Line', color='red')
        plt.xlabel('Position')
        plt.ylabel('Predicted Position')
        plt.title(f'Linear Regression Prediction of Positions for sample {i}\n')
        plt.legend()

        if save_to_disk:
            plot_path = os.path.join(output_dir, f"linear_regression_q={q}_plot.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        else:
            plt.show()

        plt.close()
    return regressor

regressors = []
for q in [0]:
    mean_curve, num_neurons = get_y(train_cache, q=0.75, layer='blocks.0.ln2.hook_normalized', draw_line=False)
    regressor = regress_samples_to_mean(train_cache, test_cache, q, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="activation_plots")
    regressors.append(regressor)


# In[ ]:


"""
Taking the strongest neurons from the regression yields pretty bad results interestingly.
"""


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Y
def get_y(test_cache, q=0.9, layer='blocks.0.ln2.hook_normalized', draw_line=True):
    neurons_avg_across_pos = train_cache[layer].mean(0).to(device)  # Move to GPU
    neurons_mean = neurons_avg_across_pos.mean(0)
    q = np.quantile(neurons_mean.cpu().numpy(), q)  # Move to CPU for numpy operations
    top_quarter_neurons_ids = np.where(neurons_mean.cpu().numpy() > q)[0]
    
    # Print the number of neurons used to create Y
    print(f"Number of neurons used to create Y: {len(top_quarter_neurons_ids)}")
    
    if draw_line:
        line(neurons_avg_across_pos[:, top_quarter_neurons_ids].cpu().numpy())  # Move to CPU for plotting
    avg_line_for_upper_quarter = neurons_avg_across_pos[:, top_quarter_neurons_ids].mean(1)
    return avg_line_for_upper_quarter.cpu().numpy(), len(top_quarter_neurons_ids)  # Return Y and number of neurons

# Build X
def regress_samples_to_mean(train_cache, test_cache, q, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="./plots_neurons"):    
    # Get activations with shape: (batch, ctx, neurons)
    activations = train_cache[act].to(device)  # Move to GPU
    activations = activations.cpu().numpy()  # Move to CPU for sklearn compatibility

    num_samples, n_positions, n_neurons = activations.shape
    print("Activations shape:", activations.shape)

    # Get selected neurons ids
    q95=np.quantile(best_regressor.coef_, 0.75)
    q5=np.quantile(best_regressor.coef_, 0.25)
    top_neurons_ids = np.where((best_regressor.coef_ < q95) & (best_regressor.coef_ > q5))[0]

    # Flatten (batch, ctx) to get X, and compute thresholds using the given quantiles.
    # Create output directory if saving plots
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Train a linear regressor for each sample
    Y_sample = mean_curve.reshape(-1,1)  # Target line [64, 1]
    regressor = LinearRegression()

    print(f'N neurons used for training: {len(top_neurons_ids)}')
    n_neurons = len(top_neurons_ids)

    selected_activations = activations[:, :, top_neurons_ids]
    Y = np.tile(Y_sample, (num_samples, 1)).reshape(-1, 1)  # Shape: (num_samples * n_positions, 1)
    regressor.fit(selected_activations.reshape(-1, n_neurons), Y)
    # Predict the positions using the trained model
    for i in [0]:
        # Extract activations for the selected neurons
        selected_activations = test_cache[act][i][:, top_neurons_ids].to(device)  # Move to GPU
        predicted_positions = regressor.predict(selected_activations.cpu().numpy())  # Move to CPU for prediction
        # Plot the true positions vs. predicted positions for the first sample
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_positions), Y_sample, label=f'True Line (Neurons: {num_neurons})', color='blue', linestyle='--')
        plt.plot(range(n_positions), predicted_positions, label='Predicted Line', color='red')
        plt.xlabel('Position')
        plt.ylabel('Predicted Position')
        plt.title(f'Linear Regression Prediction of Positions for sample {i}\n')
        plt.legend()

        if save_to_disk:
            plot_path = os.path.join(output_dir, f"linear_regression_q={q}_plot.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        else:
            plt.show()

        plt.close()
    return regressor

regressors = []
for q in [0]:
    mean_curve, num_neurons = get_y(train_cache, q=0.75, layer='blocks.0.ln2.hook_normalized', draw_line=False)
    regressor = regress_samples_to_mean(train_cache, test_cache, q, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="activation_plots")
    regressors.append(regressor)


# In[ ]:


"""
It looks like the positive and the negative neuorns cancel each other out.
"""


# # Lets take only the positive coefficients

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Y
def get_y(test_cache, q=0.9, layer='blocks.0.ln2.hook_normalized', draw_line=True):
    neurons_avg_across_pos = train_cache[layer].mean(0).to(device)  # Move to GPU
    neurons_mean = neurons_avg_across_pos.mean(0)
    q = np.quantile(neurons_mean.cpu().numpy(), q)  # Move to CPU for numpy operations
    top_quarter_neurons_ids = np.where(neurons_mean.cpu().numpy() > q)[0]
    
    # Print the number of neurons used to create Y
    print(f"Number of neurons used to create Y: {len(top_quarter_neurons_ids)}")
    
    if draw_line:
        line(neurons_avg_across_pos[:, top_quarter_neurons_ids].cpu().numpy())  # Move to CPU for plotting
    avg_line_for_upper_quarter = neurons_avg_across_pos[:, top_quarter_neurons_ids].mean(1)
    return avg_line_for_upper_quarter.cpu().numpy(), len(top_quarter_neurons_ids)  # Return Y and number of neurons

# Build X
def regress_samples_to_mean(train_cache, test_cache, q, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="./plots_neurons"):    
    # Get activations with shape: (batch, ctx, neurons)
    activations = train_cache[act].to(device)  # Move to GPU
    activations = activations.cpu().numpy()  # Move to CPU for sklearn compatibility

    num_samples, n_positions, n_neurons = activations.shape
    print("Activations shape:", activations.shape)

    # Get selected neurons ids
    q95=np.quantile(best_regressor.coef_, 0)
    top_neurons_ids = np.where((best_regressor.coef_ > q95))[0]

    # Flatten (batch, ctx) to get X, and compute thresholds using the given quantiles.
    # Create output directory if saving plots
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Train a linear regressor for each sample
    Y_sample = mean_curve.reshape(-1,1)  # Target line [64, 1]
    regressor = LinearRegression()

    print(f'N neurons used for training: {len(top_neurons_ids)}')
    n_neurons = len(top_neurons_ids)

    selected_activations = activations[:, :, top_neurons_ids]
    Y = np.tile(Y_sample, (num_samples, 1)).reshape(-1, 1)  # Shape: (num_samples * n_positions, 1)
    regressor.fit(selected_activations.reshape(-1, n_neurons), Y)
    # Predict the positions using the trained model
    for i in [0]:
        # Extract activations for the selected neurons
        selected_activations = test_cache[act][i][:, top_neurons_ids].to(device)  # Move to GPU
        predicted_positions = regressor.predict(selected_activations.cpu().numpy())  # Move to CPU for prediction
        # Plot the true positions vs. predicted positions for the first sample
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_positions), Y_sample, label=f'True Line (Neurons: {num_neurons})', color='blue', linestyle='--')
        plt.plot(range(n_positions), predicted_positions, label='Predicted Line', color='red')
        plt.xlabel('Position')
        plt.ylabel('Predicted Position')
        plt.title(f'Linear Regression Prediction of Positions for sample {i}\n')
        plt.legend()

        if save_to_disk:
            plot_path = os.path.join(output_dir, f"linear_regression_q={q}_plot.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        else:
            plt.show()

        plt.close()
    return regressor

regressors = []
for q in [0]:
    mean_curve, num_neurons = get_y(train_cache, q=0.75, layer='blocks.0.ln2.hook_normalized', draw_line=False)
    regressor = regress_samples_to_mean(train_cache, test_cache, q, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="activation_plots")
    regressors.append(regressor)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


line(regressors[-1].coef_.T)


# In[ ]:


regressors[0].coef_.shape


# In[ ]:


(regressors[0].coef_)


# ## Use only 21 neurons

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Y
def get_y(test_cache, q=0.9, layer='blocks.0.ln2.hook_normalized', draw_line=True):
    neurons_avg_across_pos = train_cache[layer].mean(0).to(device)  # Move to GPU
    neurons_mean = neurons_avg_across_pos.mean(0)
    q = np.quantile(neurons_mean.cpu().numpy(), q)  # Move to CPU for numpy operations
    top_quarter_neurons_ids = np.where(neurons_mean.cpu().numpy() > q)[0]
    
    # Print the number of neurons used to create Y
    print(f"Number of neurons used to create Y: {len(top_quarter_neurons_ids)}")
    
    if draw_line:
        line(neurons_avg_across_pos[:, top_quarter_neurons_ids].cpu().numpy())  # Move to CPU for plotting
    avg_line_for_upper_quarter = neurons_avg_across_pos[:, top_quarter_neurons_ids].mean(1)
    return avg_line_for_upper_quarter.cpu().numpy(), len(top_quarter_neurons_ids)  # Return Y and number of neurons

# Build X
def regress_samples_to_mean(train_cache, test_cache, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="./plots_neurons"):
    # Get activations with shape: (batch, ctx, neurons)
    activations = train_cache[act].to(device)  # Move to GPU
    activations = activations.cpu().numpy()  # Move to CPU for sklearn compatibility

    num_samples, n_positions, n_neurons = activations.shape
    print("Activations shape:", activations.shape)
    
    # Flatten (batch, ctx) to get X, and compute thresholds using the given quantiles.
    X = activations.reshape(-1, n_neurons)
    # Create output directory if saving plots
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Train a linear regressor for each sample
    Y_sample = mean_curve.reshape(-1,1)  # Target line [64, 1]
    regressor = LinearRegression()

    selected_activations = activations[:, :, :]
    Y = np.tile(Y_sample, (num_samples, 1)).reshape(-1, 1)  # Shape: (num_samples * n_positions, 1)
    regressor.fit(selected_activations.reshape(-1, n_neurons), Y)
    # Predict the positions using the trained model
    for i in [0]:
        # Extract activations for the selected neurons
        selected_activations = test_cache[act][i].to(device)  # Move to GPU
        predicted_positions = regressor.predict(selected_activations.cpu().numpy())  # Move to CPU for prediction
        # Plot the true positions vs. predicted positions for the first sample
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_positions), Y_sample, label=f'True Line (Neurons: {num_neurons})', color='blue', linestyle='--')
        plt.plot(range(n_positions), predicted_positions, label='Predicted Line', color='red')
        plt.xlabel('Position')
        plt.ylabel('Predicted Position')
        plt.title(f'Linear Regression Prediction of Positions for sample {i}\n')
        plt.legend()

        if save_to_disk:
            plot_path = os.path.join(output_dir, f"linear_regression_q={q}_plot.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        else:
            plt.show()

        plt.close()
    return regressor

regressors = []
for q in [0.995]:
    mean_curve, num_neurons = get_y(train_cache, q=q, layer='blocks.0.ln2.hook_normalized', draw_line=False)
    regressor = regress_samples_to_mean(train_cache, test_cache, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="activation_plots")
    regressors.append(regressor)


# In[ ]:





# ## 30K samples

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Y
def get_y(test_cache, q=0.9, layer='blocks.0.ln2.hook_normalized', draw_line=True):
    neurons_avg_across_pos = train_cache[layer].mean(0).to(device)  # Move to GPU
    neurons_mean = neurons_avg_across_pos.mean(0)
    q = np.quantile(neurons_mean.cpu().numpy(), q)  # Move to CPU for numpy operations
    top_quarter_neurons_ids = np.where(neurons_mean.cpu().numpy() > q)[0]
    
    # Print the number of neurons used to create Y
    print(f"Number of neurons used to create Y: {len(top_quarter_neurons_ids)}")
    
    if draw_line:
        line(neurons_avg_across_pos[:, top_quarter_neurons_ids].cpu().numpy())  # Move to CPU for plotting
    avg_line_for_upper_quarter = neurons_avg_across_pos[:, top_quarter_neurons_ids].mean(1)
    return avg_line_for_upper_quarter.cpu().numpy(), len(top_quarter_neurons_ids)  # Return Y and number of neurons

# Build X
def regress_samples_to_mean(train_cache, test_cache, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="./plots_neurons"):
    # Get activations with shape: (batch, ctx, neurons)
    activations = train_cache[act].to(device)  # Move to GPU
    activations = activations.cpu().numpy()  # Move to CPU for sklearn compatibility

    num_samples, n_positions, n_neurons = activations.shape
    print("Activations shape:", activations.shape)
    
    # Flatten (batch, ctx) to get X, and compute thresholds using the given quantiles.
    X = activations.reshape(-1, n_neurons)
    # Create output directory if saving plots
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Train a linear regressor for each sample
    Y_sample = mean_curve.reshape(-1,1)  # Target line [64, 1]
    regressor = LinearRegression()

    selected_activations = activations[:, :, :]
    Y = np.tile(Y_sample, (num_samples, 1)).reshape(-1, 1)  # Shape: (num_samples * n_positions, 1)
    regressor.fit(selected_activations.reshape(-1, n_neurons), Y)
    # Predict the positions using the trained model
    for i in [0]:
        # Extract activations for the selected neurons
        selected_activations = test_cache[act][i].to(device)  # Move to GPU
        predicted_positions = regressor.predict(selected_activations.cpu().numpy())  # Move to CPU for prediction
        # Plot the true positions vs. predicted positions for the first sample
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_positions), Y_sample, label=f'True Line (Neurons: {num_neurons})', color='blue', linestyle='--')
        plt.plot(range(n_positions), predicted_positions, label='Predicted Line', color='red')
        plt.xlabel('Position')
        plt.ylabel('Predicted Position')
        plt.title(f'Linear Regression Prediction of Positions for sample {i}\n')
        plt.legend()

        if save_to_disk:
            plot_path = os.path.join(output_dir, f"linear_regression_q={q}_plot.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        else:
            plt.show()

        plt.close()
    return regressor

regressors = []
for q in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
    mean_curve, num_neurons = get_y(train_cache, q=q, layer='blocks.0.ln2.hook_normalized', draw_line=False)
    regressor = regress_samples_to_mean(train_cache, test_cache, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="activation_plots")
    regressors.append(regressor)


# ## 10K samples

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Y
def get_y(test_cache, q=0.9, layer='blocks.0.ln2.hook_normalized', draw_line=True):
    neurons_avg_across_pos = train_cache[layer].mean(0).to(device)  # Move to GPU
    neurons_mean = neurons_avg_across_pos.mean(0)
    q = np.quantile(neurons_mean.cpu().numpy(), q)  # Move to CPU for numpy operations
    top_quarter_neurons_ids = np.where(neurons_mean.cpu().numpy() > q)[0]
    
    # Print the number of neurons used to create Y
    print(f"Number of neurons used to create Y: {len(top_quarter_neurons_ids)}")
    
    if draw_line:
        line(neurons_avg_across_pos[:, top_quarter_neurons_ids].cpu().numpy())  # Move to CPU for plotting
    avg_line_for_upper_quarter = neurons_avg_across_pos[:, top_quarter_neurons_ids].mean(1)
    return avg_line_for_upper_quarter.cpu().numpy(), len(top_quarter_neurons_ids)  # Return Y and number of neurons

# Build X
def regress_samples_to_mean(train_cache, test_cache, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="./plots_neurons"):
    # Get activations with shape: (batch, ctx, neurons)
    activations = train_cache[act].to(device)  # Move to GPU
    activations = activations.cpu().numpy()  # Move to CPU for sklearn compatibility

    num_samples, n_positions, n_neurons = activations.shape
    print("Activations shape:", activations.shape)
    
    # Flatten (batch, ctx) to get X, and compute thresholds using the given quantiles.
    X = activations.reshape(-1, n_neurons)
    # Create output directory if saving plots
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Train a linear regressor for each sample
    Y_sample = mean_curve.reshape(-1,1)  # Target line [64, 1]
    regressor = LinearRegression()

    selected_activations = activations[:, :, :]
    Y = np.tile(Y_sample, (num_samples, 1)).reshape(-1, 1)  # Shape: (num_samples * n_positions, 1)
    regressor.fit(selected_activations.reshape(-1, n_neurons), Y)
    # Predict the positions using the trained model
    for i in [0]:
        # Extract activations for the selected neurons
        selected_activations = test_cache[act][i].to(device)  # Move to GPU
        predicted_positions = regressor.predict(selected_activations.cpu().numpy())  # Move to CPU for prediction
        # Plot the true positions vs. predicted positions for the first sample
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_positions), Y_sample, label=f'True Line (Neurons: {num_neurons})', color='blue', linestyle='--')
        plt.plot(range(n_positions), predicted_positions, label='Predicted Line', color='red')
        plt.xlabel('Position')
        plt.ylabel('Predicted Position')
        plt.title(f'Linear Regression Prediction of Positions for sample {i}\n')
        plt.legend()

        if save_to_disk:
            plot_path = os.path.join(output_dir, f"linear_regression_q={q}_plot.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        else:
            plt.show()

        plt.close()
    return regressor

regressors = []
for q in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
    mean_curve, num_neurons = get_y(train_cache, q=q, layer='blocks.0.ln2.hook_normalized', draw_line=False)
    regressor = regress_samples_to_mean(train_cache, test_cache, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="activation_plots")
    regressors.append(regressor)


# In[ ]:


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Y
def get_y(test_cache, q=0.9, layer='blocks.0.ln2.hook_normalized', draw_line=True):
    neurons_avg_across_pos = train_cache[layer].mean(0).to(device)  # Move to GPU
    neurons_mean = neurons_avg_across_pos.mean(0)
    q = np.quantile(neurons_mean.cpu().numpy(), q)  # Move to CPU for numpy operations
    top_quarter_neurons_ids = np.where(neurons_mean.cpu().numpy() > q)[0]
    
    # Print the number of neurons used to create Y
    print(f"Number of neurons used to create Y: {len(top_quarter_neurons_ids)}")
    
    if draw_line:
        line(neurons_avg_across_pos[:, top_quarter_neurons_ids].cpu().numpy())  # Move to CPU for plotting
    avg_line_for_upper_quarter = neurons_avg_across_pos[:, top_quarter_neurons_ids].mean(1)
    return avg_line_for_upper_quarter.cpu().numpy(), len(top_quarter_neurons_ids)  # Return Y and number of neurons

# Build X
def regress_samples_to_mean(train_cache, test_cache, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="./plots_neurons"):
    # Get activations with shape: (batch, ctx, neurons)
    activations = train_cache[act].to(device)  # Move to GPU
    activations = activations.cpu().numpy()  # Move to CPU for sklearn compatibility

    num_samples, n_positions, n_neurons = activations.shape
    print("Activations shape:", activations.shape)
    
    # Flatten (batch, ctx) to get X, and compute thresholds using the given quantiles.
    X = activations.reshape(-1, n_neurons)
    # Create output directory if saving plots
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Train a linear regressor for each sample
    Y_sample = mean_curve.reshape(-1,1)  # Target line [64, 1]
    regressor = LinearRegression()

    selected_activations = activations[:, :, :]
    Y = np.tile(Y_sample, (num_samples, 1)).reshape(-1, 1)  # Shape: (num_samples * n_positions, 1)
    regressor.fit(selected_activations.reshape(-1, n_neurons), Y)
    # Predict the positions using the trained model
    for i in [0]:
        # Extract activations for the selected neurons
        selected_activations = test_cache[act][i].to(device)  # Move to GPU
        predicted_positions = regressor.predict(selected_activations.cpu().numpy())  # Move to CPU for prediction
        # Plot the true positions vs. predicted positions for the first sample
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_positions), Y_sample, label=f'True Line (Neurons: {num_neurons})', color='blue', linestyle='--')
        plt.plot(range(n_positions), predicted_positions, label='Predicted Line', color='red')
        plt.xlabel('Position')
        plt.ylabel('Predicted Position')
        plt.title(f'Linear Regression Prediction of Positions for sample {i}\n')
        plt.legend()

        if save_to_disk:
            plot_path = os.path.join(output_dir, f"linear_regression_q={q}_plot.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        else:
            plt.show()

        plt.close()
    return regressor



# # More repetitions in each sentence (smaller vocab)
# # VOCAB = 32

# In[ ]:


from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm

D_VOCAB = int(N_CTX / 2)
device='cpu'
print(f"Vocab size: {D_VOCAB}")
cfg = HookedTransformerConfig(
    n_layers=1,
    d_model=D_MODEL,
    d_head=D_MODEL,
    n_heads=1,
    d_mlp=D_MODEL*4,
    d_vocab=D_VOCAB,
    n_ctx=N_CTX,
    act_fn='relu',
    normalization_type='LNPre',
    device=device,
    use_hook_mlp_in=True
)
train_dataset = torch.randint(low=0, high=D_VOCAB, size=(N_SAMPLES, N_CTX), device=device)
test_dataset = torch.randint(low=0, high=D_VOCAB, size=(N_SAMPLES, N_CTX), device=device)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
lit_model = LitTransformer(cfg, train_loader, test_loader)
lit_model.to(device)
try:
    del train_cache, test_cache
    del model
    torch.cuda.empty_cache()
except:
    pass
model = lit_model.model
train_logits, train_cache = model.run_with_cache(train_loader.dataset)
test_logits, test_cache = model.run_with_cache(test_loader.dataset[:100])


# In[ ]:


line(train_cache['blocks.0.ln2.hook_normalized'].mean(0)[:,:100])


# In[ ]:


regressors = []
for q in [0.95]:
    mean_curve, num_neurons = get_y(train_cache, q=q, layer='blocks.0.ln2.hook_normalized', draw_line=False)
    regressor = regress_samples_to_mean(train_cache, test_cache, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="activation_plots")
    regressors.append(regressor)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from itertools import combinations

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Y
def get_y(test_cache, layer='blocks.0.ln2.hook_normalized', draw_line=True):
    neurons_avg_across_pos = train_cache[layer].mean(0).to(device)  # Move to GPU
    neurons_mean = neurons_avg_across_pos.mean(0).cpu().numpy()  # Move to CPU for numpy operations

    # Calculate pairwise distances between neurons
    pairwise_distances = np.linalg.norm(neurons_avg_across_pos.cpu().numpy()[:, :, None] - neurons_avg_across_pos.cpu().numpy()[:, None, :], axis=0)

    # Find the group of 10 neurons with the smallest average pairwise distance
    n_neurons = neurons_avg_across_pos.shape[1]
    best_group = None
    best_distance = float('inf')
    for group in combinations(range(n_neurons), 10):
        avg_distance = np.mean(pairwise_distances[np.ix_(group, group)])
        if avg_distance < best_distance:
            best_distance = avg_distance
            best_group = group

    # Use the selected group of neurons to calculate Y
    top_neurons_ids = list(best_group)
    print(f"Selected neurons for Y: {top_neurons_ids}")
    if draw_line:
        line(neurons_avg_across_pos[:, top_neurons_ids].cpu().numpy())  # Move to CPU for plotting
    avg_line_for_selected_neurons = neurons_avg_across_pos[:, top_neurons_ids].mean(1)
    return avg_line_for_selected_neurons.cpu().numpy(), len(top_neurons_ids)  # Return Y and number of neurons

# Build X
def regress_samples_to_mean(train_cache, test_cache, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="./plots_neurons"):
    # Get activations with shape: (batch, ctx, neurons)
    activations = train_cache[act].to(device)  # Move to GPU
    activations = activations.cpu().numpy()  # Move to CPU for sklearn compatibility

    num_samples, n_positions, n_neurons = activations.shape
    print("Activations shape:", activations.shape)
    
    # Flatten (batch, ctx) to get X, and compute thresholds using the given quantiles.
    X = activations.reshape(-1, n_neurons)
    # Create output directory if saving plots
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Train a linear regressor for each sample
    Y_sample = mean_curve.reshape(-1,1)  # Target line [64, 1]
    regressor = LinearRegression()

    selected_activations = activations[:, :, :]
    Y = np.tile(Y_sample, (num_samples, 1)).reshape(-1, 1)  # Shape: (num_samples * n_positions, 1)
    regressor.fit(selected_activations.reshape(-1, n_neurons), Y)
    # Predict the positions using the trained model
    for i in [0]:
        # Extract activations for the selected neurons
        selected_activations = test_cache[act][i].to(device)  # Move to GPU
        predicted_positions = regressor.predict(selected_activations.cpu().numpy())  # Move to CPU for prediction
        # Plot the true positions vs. predicted positions for the first sample
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_positions), Y_sample, label=f'True Line (Neurons: {num_neurons})', color='blue', linestyle='--')
        plt.plot(range(n_positions), predicted_positions, label='Predicted Line', color='red')
        plt.xlabel('Position')
        plt.ylabel('Predicted Position')
        plt.title(f'Linear Regression Prediction of Positions for sample {i}\n')
        plt.legend()

        if save_to_disk:
            plot_path = os.path.join(output_dir, f"linear_regression_plot.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        else:
            plt.show()

        plt.close()
    return regressor

# Example call:
regressors = []
for layer in ['blocks.0.ln2.hook_normalized']:
    mean_curve, num_neurons = get_y(train_cache, layer=layer, draw_line=False)
    regressor = regress_samples_to_mean(train_cache, test_cache, mean_curve, num_neurons, act=layer, save_to_disk=False, output_dir="activation_plots")
    regressors.append(regressor)


# # Calculate the closest set of 100 Neurons in terms of MAE

# In[ ]:


import random
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from itertools import combinations

N_NEURONS_FOR_Y_CALC = 200
# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Y: using average over samples then selecting 10 neurons with minimal average MAE
def get_y(test_cache, layer='blocks.0.ln2.hook_normalized', draw_line=True):
    # Average over samples: result shape [pos, neurons]
    neurons_avg_across_pos = train_cache[layer].mean(0).to(device)  # [pos, neurons]
    neurons_avg_across_pos_cpu = neurons_avg_across_pos.cpu().numpy()  # shape [pos, n_neurons]
    # Filter out neurons whose mean over positions is not positive.
    positive_mask = neurons_avg_across_pos_cpu.mean(axis=0) > 0
    neurons_avg_across_pos_cpu = neurons_avg_across_pos_cpu[:, positive_mask]
    if not neurons_avg_across_pos_cpu.shape[1]:
        raise ValueError("No neurons with positive average activation found.")
    
    pos, n_neurons = neurons_avg_across_pos_cpu.shape

    # Precompute the distance (MAE over positions) between each pair of neurons.
    # distance_matrix[i, j] = mean(|neurons_avg_across_pos[:, i] - neurons_avg_across_pos[:, j]|)
    distance_matrix = np.zeros((n_neurons, n_neurons))
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            mae = np.mean(np.abs(neurons_avg_across_pos_cpu[:, i] - neurons_avg_across_pos_cpu[:, j]))
            distance_matrix[i, j] = mae

    # Find the combination of 10 neurons with minimal average pairwise MAE.
    best_group = None
    best_avg_mae = float('inf')
    max_iter = 1_000_000
    searched_groups = set()  # Track already seen groups
    for _ in tqdm(range(max_iter), desc="Uniformly sampling neuron groups"):
        group = tuple(sorted(random.sample(range(n_neurons), N_NEURONS_FOR_Y_CALC)))
        if group in searched_groups:
            continue
        searched_groups.add(group)
        submatrix = distance_matrix[np.ix_(group, group)]
        total_mae = submatrix.sum()
        if total_mae < best_avg_mae:
            best_avg_mae = total_mae
            best_group = group

    top_neurons_ids = list(best_group)
    print(f"Selected neurons for Y: {top_neurons_ids}")
    if draw_line:
        # Plot the activations for the selected neurons; shape [pos, 10]
        line(neurons_avg_across_pos_cpu[:, top_neurons_ids])
        plt.show()
    # Compute the average line (Y) over the chosen neurons (axis=1 gives [pos])
    avg_line_for_selected_neurons = neurons_avg_across_pos_cpu[:, top_neurons_ids].mean(axis=1)
    return avg_line_for_selected_neurons, len(top_neurons_ids)


# Build X remains the same (using the regressor to predict positions)
def regress_samples_to_mean(train_cache, test_cache, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="./plots_neurons"):
    # Get activations with shape: (batch, pos, neurons)
    activations = train_cache[act].to(device)
    activations = activations.cpu().numpy()

    num_samples, n_positions, n_neurons_total = activations.shape
    print("Activations shape:", activations.shape)
    
    # Flatten (batch, pos) such that X shape is (num_samples * pos, n_neurons_total)
    X = activations.reshape(-1, n_neurons_total)
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Setup target Y_sample from our computed mean_curve (shape [pos,])
    Y_sample = mean_curve.reshape(-1, 1)
    regressor = LinearRegression()
    
    # Repeat Y_sample over samples to match X's first dimension:
    Y = np.tile(Y_sample, (num_samples, 1))
    regressor.fit(X, Y)
    
    # Predict on test set for first sample at least
    for i in [0]:
        selected_activations = test_cache[act][i].to(device)
        predicted_positions = regressor.predict(selected_activations.cpu().numpy())
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_positions), Y_sample, label=f'True Line (Neurons: {num_neurons})', color='blue', linestyle='--')
        plt.plot(range(n_positions), predicted_positions, label='Predicted Line', color='red')
        plt.xlabel('Position')
        plt.ylabel('Predicted Position')
        plt.title(f'Linear Regression Prediction of Positions for sample {i}\n')
        plt.legend()
        if save_to_disk:
            plot_path = os.path.join(output_dir, f"linear_regression_plot.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        else:
            plt.show()
        plt.close()
    return regressor

# Example call:
regressors = []
mean_curve, num_neurons = get_y(train_cache, layer='blocks.0.ln2.hook_normalized', draw_line=False)
regressor = regress_samples_to_mean(train_cache, test_cache, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="activation_plots")
regressors.append(regressor)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 1 sample to mean

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Y
def get_y(test_cache, q=0.9, layer='blocks.0.ln2.hook_normalized', draw_line=True):
    neurons_avg_across_pos = train_cache[layer].mean(0).to(device)  # Move to GPU
    neurons_mean = neurons_avg_across_pos.mean(0)
    q = np.quantile(neurons_mean.cpu().numpy(), q)  # Move to CPU for numpy operations
    top_quarter_neurons_ids = np.where(neurons_mean.cpu().numpy() > q)[0]
    
    # Print the number of neurons used to create Y
    print(f"Number of neurons used to create Y: {len(top_quarter_neurons_ids)}")
    
    if draw_line:
        line(neurons_avg_across_pos[:, top_quarter_neurons_ids].cpu().numpy())  # Move to CPU for plotting
    avg_line_for_upper_quarter = neurons_avg_across_pos[:, top_quarter_neurons_ids].mean(1)
    return avg_line_for_upper_quarter.cpu().numpy(), len(top_quarter_neurons_ids)  # Return Y and number of neurons

# Build X
def regress_samples_to_mean(train_cache, test_cache, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="./plots_neurons"):
    # Get activations with shape: (batch, ctx, neurons)
    activations = train_cache[act].to(device)  # Move to GPU
    activations = activations.cpu().numpy()  # Move to CPU for sklearn compatibility

    num_samples, n_positions, n_neurons = activations.shape
    print("Activations shape:", activations.shape)
    
    # Flatten (batch, ctx) to get X, and compute thresholds using the given quantiles.
    X = activations.reshape(-1, n_neurons)
    # Create output directory if saving plots
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Train a linear regressor for each sample
    Y_sample = mean_curve.reshape(-1,1)  # Target line [64, 1]
    regressor = LinearRegression()

    selected_activations = activations[:, :, :]
    Y = np.tile(Y_sample, (num_samples, 1)).reshape(-1, 1)  # Shape: (num_samples * n_positions, 1)
    regressor.fit(selected_activations.reshape(-1, n_neurons), Y)
    # Predict the positions using the trained model
    for i in [0]:
        # Extract activations for the selected neurons
        selected_activations = test_cache[act][i].to(device)  # Move to GPU
        predicted_positions = regressor.predict(selected_activations.cpu().numpy())  # Move to CPU for prediction
        # Plot the true positions vs. predicted positions for the first sample
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_positions), Y_sample, label=f'True Line (Neurons: {num_neurons})', color='blue', linestyle='--')
        plt.plot(range(n_positions), predicted_positions, label='Predicted Line', color='red')
        plt.xlabel('Position')
        plt.ylabel('Predicted Position')
        plt.title(f'Linear Regression Prediction of Positions for sample {i}\n')
        plt.legend()

        if save_to_disk:
            plot_path = os.path.join(output_dir, f"linear_regression_q={q}_plot.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        else:
            plt.show()

        plt.close()
    return regressor

# Example call:
regressors = []
for q in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
    mean_curve, num_neurons = get_y(train_cache, q=q, layer='blocks.0.ln2.hook_normalized', draw_line=False)
    regressor = regress_samples_to_mean(train_cache, test_cache, mean_curve, num_neurons, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="activation_plots")
    regressors.append(regressor)


# In[ ]:





# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Y
def get_y(test_cache, q=0.9, layer='blocks.0.ln2.hook_normalized', draw_line=True):
    neurons_avg_across_pos = train_cache[layer].mean(0).to(device)  # Move to GPU
    neurons_mean = neurons_avg_across_pos.mean(0)
    q = np.quantile(neurons_mean.cpu().numpy(), q)  # Move to CPU for numpy operations
    top_quarter_neurons_ids = np.where(neurons_mean.cpu().numpy() > q)[0]
    
    # Print the number of neurons used to create Y
    print(f"Number of neurons used to create Y: {len(top_quarter_neurons_ids)}")
    
    if draw_line:
        line(neurons_avg_across_pos[:, top_quarter_neurons_ids].cpu().numpy())  # Move to CPU for plotting
    avg_line_for_upper_quarter = neurons_avg_across_pos[:, top_quarter_neurons_ids].mean(1)
    return avg_line_for_upper_quarter.cpu().numpy()  # Return to CPU for further processing

# Build X
def regress_samples_to_mean(train_cache, test_cache, mean_curve, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="./plots_neurons"):
    # Get activations with shape: (batch, ctx, neurons)
    activations = train_cache[act].to(device)  # Move to GPU
    activations = activations.cpu().numpy()  # Move to CPU for sklearn compatibility

    num_samples, n_positions, n_neurons = activations.shape
    print("Activations shape:", activations.shape)
    
    # Flatten (batch, ctx) to get X, and compute thresholds using the given quantiles.
    X = activations.reshape(-1, n_neurons)
    # Create output directory if saving plots
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Train a linear regressor for each sample
    Y_sample = mean_curve.reshape(-1,1)  # Target line [64, 1]
    regressor = LinearRegression()

    selected_activations = activations[:, :, :]
    Y = np.tile(Y_sample, (num_samples, 1)).reshape(-1, 1)  # Shape: (num_samples * n_positions, 1)
    regressor.fit(selected_activations.reshape(-1, n_neurons), Y)
    # Predict the positions using the trained model
    for i in [0]:
        # Extract activations for the selected neurons
        selected_activations = test_cache[act][i].to(device)  # Move to GPU
        predicted_positions = regressor.predict(selected_activations.cpu().numpy())  # Move to CPU for prediction
        # Plot the true positions vs. predicted positions for the first sample
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_positions), Y_sample, label='True Line', color='blue', linestyle='--')
        plt.plot(range(n_positions), predicted_positions, label='Predicted Line', color='red')
        plt.xlabel('Position')
        plt.ylabel('Predicted Position')
        plt.title(f'Linear Regression Prediction of Positions for sample {i}\n')
        plt.legend()

        if save_to_disk:
            plot_path = os.path.join(output_dir, f"linear_regression_q={q}_plot.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        else:
            plt.show()

        plt.close()
    return regressor

# Example call:
regressors = []
for q in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
    mean_curve = get_y(train_cache, q=q, layer='blocks.0.ln2.hook_normalized', draw_line=False)
    regressor = regress_samples_to_mean(train_cache, test_cache, mean_curve, act='blocks.0.ln2.hook_normalized', save_to_disk=False, output_dir="activation_plots")
    regressors.append(regressor)


# In[ ]:


# More repetitions in each sentence (smaller vocab)
# VOCAB = 512


# In[ ]:


D_VOCAB


# In[ ]:


D_VOCAB = int(N_CTX / 2)
print(f"Vocab size: {D_VOCAB}")
cfg = HookedTransformerConfig(
    n_layers=1,
    d_model=D_MODEL,
    d_head=D_MODEL,
    n_heads=1,
    d_mlp=D_MODEL*4,
    d_vocab=D_VOCAB,
    n_ctx=N_CTX,
    act_fn='relu',
    normalization_type='LNPre',
    device=device,
    use_hook_mlp_in=True
)

lit_model = LitTransformer(cfg, train_loader, test_loader)
lit_model.to(device)
model = lit_model.model
train_logits, train_cache = model.run_with_cache(train_loader.dataset)
test_logits, test_cache = model.run_with_cache(test_loader.dataset[:100])


# In[ ]:





# In[ ]:





# In[ ]:


# Now let's find the weight for each neuron to get into this line
avg_line_for_upper_quarter_broadcasted = np.tile(avg_line_for_upper_quarter[:, np.newaxis], (1, len(top_quarter_neurons_ids)))
avg_line_for_upper_quarter_broadcasted.shape


# In[ ]:


neurons_weight = neurons_avg_across_pos[:, top_quarter_neurons_ids] / avg_line_for_upper_quarter_broadcasted 
line(neurons_weight) 


# In[ ]:





# In[ ]:


np.quantile(cache['blocks.0.ln2.hook_normalized'].mean(0), 0.75, axis=0)


# In[ ]:


help(np.quantile)


# In[ ]:


cache


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# L0
n_layers = 1
for layer_id in range(n_layers):
    # Heads
    for i in range(cfg.n_heads):
        imshow(
            (np.exp(cache[f"blocks.{layer_id}.attn.hook_attn_scores"][:, i, :, :].mean([0]).cpu().numpy())),
            title=f"Layer {layer_id} Attention Pattern {i}",
            height=500,
            width=500,
        )


# In[ ]:


# L0
n_layers = 1
for layer_id in range(n_layers):
    # Heads
    for i in range(cfg.n_heads):
        imshow(
            to_numpy(cache["attn", layer_id][:, [i], :, :].mean([0, 1])),
            # title=f"Layer {layer_id} Attention Pattern {i}",
            height=500,
            width=500,
        )


# In[ ]:


line(cache['blocks.0.attn.hook_pattern'].mean(0).squeeze(0).var(-1), xaxis='Position', yaxis='Variance')


# In[ ]:


line(cache['blocks.0.attn.hook_pattern'][0].squeeze(0).var(-1), xaxis='Position', yaxis='Variance')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden = self.relu(self.hidden_layer(x))
        output = self.output_layer(hidden)
        return output

# ---------------- Improved Activation Statistics Plotting ----------------
def plot_activation_statistics_improved(act='blocks.0.ln2.hook_normalized', q=0.1, save_to_disk=False, output_dir="./plots_neurons"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Get activations with shape: (batch, ctx, neurons)
    train_activations = train_cache[act]
    train_activations = train_activations .cpu().numpy()

    test_activations = test_cache[act]
    test_activations = test_activations.cpu().numpy()

    num_samples, n_positions, n_neurons = train_activations.shape
    print("Train Activations shape:", train_activations.shape)
    print("Test Activations shape:", test_activations.shape)
    
    X = train_activations.reshape(-1, n_neurons)

    selected_neurons_ids = np.arange(n_neurons)

    # Create output directory if saving plots
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare data for training
    range_vals = np.arange(1, n_positions + 1).reshape(-1, 1)  # Target line [1, 64]
    selected_activations = train_activations[:, :, selected_neurons_ids]
    X_train = selected_activations.reshape(-1, len(selected_neurons_ids))  # Shape: (num_samples * n_positions, len(selected_neurons_ids))
    Y_train = np.tile(range_vals, (num_samples, 1)).reshape(-1, 1)  # Shape: (num_samples * n_positions, 1)

    # Convert data to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)

    # Define the MLP model
    input_size = len(selected_neurons_ids)
    hidden_size = 128  # You can adjust this
    output_size = 1
    model = MLP(input_size, hidden_size, output_size).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the MLP
    num_epochs = 10_000
    l1_lambda = 1
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        l1_penalty = l1_lambda * torch.sum(torch.abs(model.hidden_layer.weight))
        loss += l1_penalty
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Predict the positions using the trained model
    for i in [0, 1, 2, 3]:
        # Extract activations for the selected neurons
        selected_activations = test_activations[i][:, selected_neurons_ids]
        X_test = torch.tensor(selected_activations, dtype=torch.float32).to(device)
        predicted_positions = model(X_test).detach().cpu().numpy()

        # Plot the true positions vs. predicted positions for the first sample
        if i in [0, 1, 2, 3]:
            plt.figure(figsize=(10, 6))
            plt.plot(range_vals, range_vals, label='True Line', color='blue', linestyle='--')
            plt.plot(range_vals, predicted_positions, label='Predicted Line', color='red')
            plt.xlabel('Position')
            plt.ylabel('Predicted Position')
            plt.title(f'MLP Prediction of Positions for sample {i}')
            plt.legend()

            if save_to_disk:
                plot_path = os.path.join(output_dir, f"mlp_q={q}_plot.png")
                plt.savefig(plot_path)
                print(f"Plot saved to {plot_path}")
            else:
                plt.show()

            plt.close()

    return model, selected_neurons_ids
# Example call:
model, seleceted_neurons_ids = plot_activation_statistics_improved(act='blocks.0.ln2.hook_normalized', q=0.25, save_to_disk=False, output_dir="activation_plots")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


plot_activation_statistics_improved(act='blocks.0.ln2.hook_normalized', q=0.01, save_to_disk=False, output_dir="activation_plots")


# In[ ]:





# # Find best subgroup

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations
from tqdm import tqdm

SAMPLE = None

# ---------------- Improved Activation Statistics Plotting ----------------
def plot_activation_statistics_improved(act='blocks.0.ln2.hook_normalized', q=0.1, save_to_disk=False, output_dir="./plots_neurons"):
    # Get activations with shape: (batch, ctx, neurons)
    activations = cache[act]
    # activations = activations[range(SAMPLE)]
    activations = activations.cpu().numpy()

    num_samples, n_positions, n_neurons = activations.shape
    print("Activations shape:", activations.shape)
    
    # Flatten (batch, ctx) to get X, and compute thresholds using the given quantiles.
    X = activations.reshape(-1, n_neurons)
    neurons_mean_val = X.mean(axis=0) 
    positive_neurons = neurons_mean_val[neurons_mean_val > 0]
    top_10_percent = np.quantile(positive_neurons, 1 - q)  # Returns a scalar
    avg_top10_neurons_mask = neurons_mean_val > top_10_percent
    avg_top10_neurons_ids = np.where(avg_top10_neurons_mask)[0] 

    # Create output directory if saving plots
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize variables to track correlations
    subgroup_correlations = {}  # Dictionary to store correlations for each subgroup

    # Iterate over all subgroup sizes
    for r in tqdm(range(1, len(avg_top10_neurons_ids) + 1)):
        if r > 3 and r < len(avg_top10_neurons_ids) - 2:
            continue
        for subgroup in tqdm(combinations(avg_top10_neurons_ids, r)):  # Generate all combinations of size `r`
            subgroup = tuple(subgroup)  # Convert to tuple to use as a dictionary key
            subgroup_correlations[subgroup] = []  # Initialize list to store correlations for this subgroup

            # Calculate correlation for each sample
            for i in range(num_samples):
                subgroup_mean = activations[i][:, subgroup].mean(axis=1)  # Mean over all positions for this subgroup
                range_vals = np.arange(1, n_positions + 1)  # Range to correlate with
                corr = np.corrcoef(subgroup_mean, range_vals)[0, 1]
                subgroup_correlations[subgroup].append(corr)

    # Calculate the average correlation for each subgroup across all samples
    avg_correlations = {subgroup: np.mean(corrs) for subgroup, corrs in subgroup_correlations.items()}

    # Find the subgroup with the highest average correlation
    best_subgroup = max(avg_correlations, key=avg_correlations.get)
    best_avg_corr = avg_correlations[best_subgroup]

    print(f"Best subgroup: {best_subgroup}, Average correlation: {best_avg_corr}")

    # Plot the best subgroup for the first sample
    best_subgroup_mean = activations[0][:, best_subgroup].mean(axis=1)
    positions = np.arange(n_positions)
    plt.figure(figsize=(10, 6))
    plt.plot(positions, best_subgroup_mean, marker='^', label='Best Subgroup', color='green')
    plt.xlabel('Position')
    plt.ylabel('Activation Mean over neurons: {best_subgroup}')
    plt.title(f'Best Subgroup Correlation Plot (Average Corr: {best_avg_corr:.4f})')
    plt.legend()

    if save_to_disk:
        # Save the plot to disk
        plot_path = os.path.join(output_dir, f"best_subgroup_q={q}_plot.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    else:
        # Show the plot
        plt.show()

    plt.close()

# Example call:
# plot_activation_statistics_improved(act='blocks.0.ln2.hook_normalized', q=0.03, save_to_disk=False, output_dir="activation_plots")


# In[ ]:


for i in [0.01,0.03,0.05,0.08,0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    plot_activation_statistics_improved(act='blocks.0.ln2.hook_normalized', q=i,  save_to_disk=True, output_dir="./plots_neurons")
    print(i)
    print("Done")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




