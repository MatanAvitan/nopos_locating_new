#!/usr/bin/env python
# coding: utf-8

# In[24]:


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
    fig = px.line(tensor, **kwargs)

    showlegend=True if legend_title != "" else False

    # Update labels explicitly for x-axis and y-axis
    fig.update_layout(
        title=title,
        xaxis_title=xaxis,  # Set x-axis label
        yaxis_title=yaxis,  # Set y-axis label
        height=400,  # Optional: maintain height for consistency
        showlegend=showlegend,
        legend_title=legend_title  # Set legend title
    )
    
    # Apply line labels if provided
    if line_labels:
        for c, label in enumerate(line_labels):
            fig.data[c].name = label
    
    fig.show()

    

def imshow(tensor, yaxis="", xaxis="", **kwargs):
    tensor = to_numpy(tensor)
    plot_kwargs = {
        "color_continuous_scale": "RdBu",
        "color_continuous_midpoint": 0.0,
        "labels": {"x": xaxis, "y": yaxis},
    }
    plot_kwargs.update(kwargs)
    px.imshow(tensor, **plot_kwargs).show()


# # Load trained model

# In[2]:


device='cpu'
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
        # print(self.model.pos_embed.W_pos.data)
        # print(self.model.blocks[0].attn.W_Q.data)
        # print(self.model.W_E.device)
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
D_VOCAB = 1_024
D_MODEL = 1_024
N_SAMPLES = D_VOCAB*10
BATCH_SIZE = N_SAMPLES 
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
    std = np.sqrt(1.0 / (x.shape[1]))
    return nn.init.normal_(x, mean=0.0, std=std)

lit_model = LitTransformer(cfg, train_loader, test_loader)
lit_model.to(device)
model = lit_model.model


# In[ ]:


model.embed.W_E.data.std()


# In[ ]:


model.blocks[0].attn.W_Q.data.std()


# In[ ]:


update_embedding(model.embed.W_E)


# In[ ]:


model.embed.W_E.data.std()


# In[ ]:


model.blocks[0].attn.W_Q.data.std()


# In[ ]:


logits, cache = model.run_with_cache(test_loader.dataset)
logits.shape




import numpy as np
import matplotlib.pyplot as plt

# ---------------- Improved Activation Statistics Plotting ----------------
def plot_activation_statistics_improved(act='blocks.0.ln2.hook_normalized', q=0.1):
    # Get activations with shape: (batch, ctx, neurons)
    activations = cache[act]
    activations = activations.cpu().numpy()

    num_samples, n_positions, n_neurons = activations.shape
    print("Activations shape:", activations.shape)
    
    # Flatten (batch, ctx) to get X, and compute thresholds using the given quantiles.
    X = activations.reshape(-1, n_neurons)
    neurons_mean_val = X.mean(axis=0) 
    positive_neurons =  neurons_mean_val[neurons_mean_val>0]
    negative_neurons =  neurons_mean_val[neurons_mean_val<0]
    top_10_percent = np.quantile(positive_neurons, 1-q) # Returns a scalar
    bottom_10_percent = np.quantile(negative_neurons, q)
    print(top_10_percent, bottom_10_percent)
    avg_top10_neurons_mask = neurons_mean_val>top_10_percent
    avg_bottom10_neurons_mask = neurons_mean_val<bottom_10_percent
    # For each sample, compute per-position averages
    for i in range(num_samples):
        sample = activations[i]  # shape: (n_positions, n_neurons)
        positions = np.arange(n_positions)
    
        avg_all = []          # Mean over all neurons.
        avg_mod = []          # Mean over neurons with |x| in [lower_thresh, upper_thresh]
        avg_pos_mod = []      # Mean over positive neurons in moderate range.
        avg_neg_mod = []      # Mean over negative neurons in moderate range.
        avg_top_q_pos = []   # Mean over top 30% positive neurons.
        avg_bottom_q_neg = []  # Mean over bottom 30% negative neurons.
        avg_abs_in_q = []

        for pos_vec in sample:
            position_top10neurons_avg = pos_vec[avg_top10_neurons_mask].mean()
            position_bottom10neurons_avg =  pos_vec[avg_bottom10_neurons_mask].mean()
            abs_in_q =  pos_vec[avg_top10_neurons_mask | avg_bottom10_neurons_mask].mean()

            avg_all.append(pos_vec.mean())
            avg_top_q_pos.append(position_top10neurons_avg)
            avg_bottom_q_neg.append(position_bottom10neurons_avg)
            avg_abs_in_q.append(abs_in_q)

        avg_all = np.array(avg_all)
        # avg_mod = np.array(avg_mod)
        # avg_pos_mod = np.array(avg_pos_mod)
        # avg_neg_mod = np.array(avg_neg_mod)
        avg_top_q_pos = np.array(avg_top_q_pos)
        avg_bottom_q_neg = np.array(avg_bottom_q_neg)
        avg_abs_in_q = np.array(avg_abs_in_q)
    
        if i % 5_000 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(positions, avg_all, marker='o', label='Mean (All Neurons)')
            # plt.plot(positions, avg_pos_mod, marker='^', label='Mean (Positive Moderates)')
            # plt.plot(positions, avg_neg_mod, marker='v', label='Mean (Negative Moderates)')
            plt.plot(positions, avg_top_q_pos, marker='s', label=f'Mean (Top {q} Positive)', color='orange')
            plt.plot(positions, avg_bottom_q_neg, marker='d', label=f'Mean (Bottom {q}% Negative)', color='purple')
            plt.plot(positions, avg_abs_in_q, marker='x', label=f'Val > {top_10_percent} or < {bottom_10_percent}', color='red')
            plt.xlabel('Position')
            plt.ylabel('Activation Mean')
            plt.title(f'Sample {i}: Average Neuron Activations per Position')
            plt.legend()
            plt.show()
            print(f"Plotted sample {i}")

# # Save to disk!

# ## At the sample level


import numpy as np
import matplotlib.pyplot as plt
import os

def plot_activation_statistics_improved(act='blocks.0.ln2.hook_normalized', q=0.1, save_to_disk=False, output_dir="plots"):
    # Get activations with shape: (batch, ctx, neurons)
    activations = cache[act]
    activations = activations.cpu().numpy()

    num_samples, n_positions, n_neurons = activations.shape
    print("Activations shape:", activations.shape)
    
    # Flatten (batch, ctx) to get X, and compute thresholds using the given quantiles.
    X = activations.reshape(-1, n_neurons)
    neurons_mean_val = X.mean(axis=0) 
    positive_neurons = neurons_mean_val[neurons_mean_val > 0]
    negative_neurons = neurons_mean_val[neurons_mean_val < 0]
    top_10_percent = np.quantile(positive_neurons, 1 - q)  # Returns a scalar
    bottom_10_percent = np.quantile(negative_neurons, q)
    print(top_10_percent, bottom_10_percent)
    avg_top10_neurons_mask = neurons_mean_val > top_10_percent
    avg_bottom10_neurons_mask = neurons_mean_val < bottom_10_percent

    # Create output directory if saving plots
    if save_to_disk and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # For each sample, compute per-position averages
    for i in range(num_samples):
        sample = activations[i]  # shape: (n_positions, n_neurons)
        positions = np.arange(n_positions)
    
        avg_all = []          # Mean over all neurons.
        avg_top_q_pos = []    # Mean over top 10% positive neurons.
        avg_bottom_q_neg = [] # Mean over bottom 10% negative neurons.
        avg_abs_in_q = []

        for pos_vec in sample:
            position_top10neurons_avg = pos_vec[avg_top10_neurons_mask].mean()
            position_bottom10neurons_avg = pos_vec[avg_bottom10_neurons_mask].mean()
            abs_in_q = pos_vec[avg_top10_neurons_mask].mean() - pos_vec[avg_bottom10_neurons_mask].mean()

            avg_all.append(pos_vec.mean())
            avg_top_q_pos.append(position_top10neurons_avg)
            avg_bottom_q_neg.append(position_bottom10neurons_avg)
            avg_abs_in_q.append(abs_in_q)

        avg_all = np.array(avg_all)
        avg_top_q_pos = np.array(avg_top_q_pos)
        avg_bottom_q_neg = np.array(avg_bottom_q_neg)
        avg_abs_in_q = np.array(avg_abs_in_q)
    
        if i == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(positions, avg_all, marker='o', label='Mean (All Neurons)')
            plt.plot(positions, avg_top_q_pos, marker='s', label=f'Mean (Top {q} Positive)', color='orange')
            plt.plot(positions, avg_bottom_q_neg, marker='d', label=f'Mean (Bottom {q}% Negative)', color='purple')
            plt.plot(positions, avg_abs_in_q, marker='x', label=f'Val > {top_10_percent} or < {bottom_10_percent}', color='red')
            plt.xlabel('Position')
            plt.ylabel('Activation Mean')
            plt.title(f'Sample {i}: Average Neuron Activations per Position')
            plt.legend()

            if save_to_disk:
                # Save the plot to disk
                plot_path = os.path.join(output_dir, f"sample_{i}_q={q}_plot.png")
                plt.savefig(plot_path)
                print(f"Plot saved to {plot_path}")
            else:
                # Show the plot
                plt.show()

            plt.close()

# Example call:
# plot_activation_statistics_improved(act='blocks.0.ln2.hook_normalized', q=0.1, save_to_disk=False, output_dir="./activation_plots")


# In[ ]:


# for i in [0.01,0.03,0.05,0.08,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]:
    # plot_activation_statistics_improved(act='blocks.0.ln2.hook_normalized', q=i,  save_to_disk=True)
    # print(i)
    # print("Done")


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
        if r>4:
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
plot_activation_statistics_improved(act='blocks.0.ln2.hook_normalized', q=0.03, save_to_disk=False, output_dir="activation_plots")



for i in [0.01,0.03,0.05,0.08,0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    print('here')
    plot_activation_statistics_improved(act='blocks.0.ln2.hook_normalized', q=i,  save_to_disk=True, output_dir="./plots_neurons")
    print(i)
    print("Done")



