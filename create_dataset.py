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
import pytorch_lightning as pl
from functools import partial
import pandas as pd
from utils import device

D_VOCAB=64
torch.manual_seed(0)

def make_data_generator(cfg, batch_size):
    while True:
        ctx = cfg.n_ctx         # Number of elements in each sample     
        # Generate random values for each sample
        samples = []
        for _ in range(batch_size):
            # There are D_VOCAB-1 options, as token 0 stands for a separator.
            sample = np.random.choice(np.arange(1, D_VOCAB), size=ctx, replace=True)
            samples.append(sample)
        
        # Convert the list of samples to a NumPy array
        x = np.array(samples)
        query_pos = np.random.choice(np.arange(2, ctx), size=x.shape[0], replace=True) # Pos 0 stores the pos until to sum and pos 1 is the separator token.
        x[:, 0] = query_pos
        x[:, 1] = 0
        yield torch.tensor(x)

def create_dataset(data_generator, n_batches):
    ds_set = set()
    for i in range(n_batches):
        batch = next(data_generator)
        batch_as_list = batch.tolist()
        batch_as_distinct_set = set(map(tuple, batch_as_list))
        for sample in batch_as_distinct_set:
            ds_set.add(sample)
        
    # Convert to tensor
    ds_tensor = torch.stack([torch.tensor(batch) for batch in ds_set]).reshape(-1, cfg.n_ctx)
    return ds_tensor

def split_train_test(ds_tensor, train_ratio):
    train_max_idx = int(train_ratio * ds_tensor.shape[0])
    train_tensor = ds_tensor[:train_max_idx]
    test_tensor = ds_tensor[train_max_idx:]
    return train_tensor, test_tensor

def build_input_and_labels(tokens):
    tokens = tokens.to(device)
    targets = torch.zeros(tokens.shape[0], dtype=int, device=device)
    batch_size = tokens.shape[0]
   
    query_tokens = tokens[:, 0]
    seq_start_idx = 2

    # Extract the values from query_tokens as integers
    # query_tokens_int = query_tokens.tolist()

    # Update tokens using the extracted integers
    for i in range(batch_size):
        axis_1_idxs = torch.arange(start=2, end=(query_tokens[i] + 1), step=2)
        targets[i] = tokens[i, axis_1_idxs].sum()

    return tokens, targets


if IS_FIRST:
    data_generator = make_data_generator(cfg, BATCH_SIZE)
    ds_tensor = create_dataset(data_generator, N_BATCHES)
    train_tensor, test_tensor = split_train_test(ds_tensor, TRAIN_RATIO)
    train_tensor = train_tensor.to(device)
    test_tensor = test_tensor.to(device)
    
    train_tokens, train_targets = build_input_and_labels(train_tensor)
    test_tokens, test_targets = build_input_and_labels(test_tensor)
    
    print(train_tensor.shape, test_tensor.shape)
    train_dataset, test_dataset = TensorDataset(train_tokens, train_targets), TensorDataset(test_tokens, test_targets)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    exp_dir = Path(EXP_NAME)
    exp_dir.mkdir(exist_ok=True)
    torch.save(train_dataset, exp_dir / 'train_dataset')
    torch.save(test_dataset, exp_dir / 'test_dataset')
else:
    exp_dir = Path(EXP_NAME)
    train_dataset = torch.load(exp_dir / 'train_dataset', map_location=device)
    test_dataset = torch.load(exp_dir / 'test_dataset', map_location=device)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

def evaluate(model, test_loader, device):
    # model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for tokens, targets in tqdm.tqdm(test_loader, desc="Evaluating"):
            # Forward pass
            logits = model(tokens)
            # Calculate loss
            axis_0_idxs = torch.arange(tokens.shape[0], device=device)
            loss = loss_fn(logits[axis_0_idxs, tokens[:, 0], :], targets)
            total_loss += loss.item()

            # Convert logits to predicted labels (assuming you are using softmax)
            predicted_labels = logits[axis_0_idxs, tokens[:, 0], :].argmax(dim=-1)

            # Append targets and predictions for accuracy calculation
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted_labels.cpu().numpy())

    # Calculate accuracy
    all_predictions_arr, all_targets_arr = np.array(all_predictions), np.array(all_targets)
    n_samples = all_predictions_arr.shape[0]
    accuracy = (all_predictions_arr == all_targets_arr).sum() / n_samples

    # Calculate average loss
    average_loss = total_loss / len(test_loader)

    return accuracy, average_loss