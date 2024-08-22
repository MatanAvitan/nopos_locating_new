import torch
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from tqdm.auto import tqdm
from pathlib import Path
from utils import device

OUTPUT_DIR = Path('datasets/abs_pos_pred_random_values')
D_VOCAB = 64
CTX = 64
torch.manual_seed(0)

class CustomDataset(TensorDataset):
    def __init__(self, tokens, targets):
        super().__init__(tokens, targets)

def make_data_generator(batch_size):
    while True:
        samples = []
        for _ in range(batch_size):
            sample = np.random.choice(np.arange(D_VOCAB), size=CTX, replace=True)
            samples.append(sample)
        x = np.array(samples)
        yield torch.tensor(x)

def create_dataset(data_generator, n_batches):
    ds_set = set()
    for _ in range(n_batches):
        batch = next(data_generator)
        batch_as_list = batch.tolist()
        ds_set.update(map(tuple, batch_as_list))
        
    ds_tensor = torch.stack([torch.tensor(sample) for sample in ds_set]).reshape(-1, CTX)
    return ds_tensor

def split_train_test(ds_tensor, train_ratio):
    train_max_idx = int(train_ratio * ds_tensor.shape[0])
    return ds_tensor[:train_max_idx], ds_tensor[train_max_idx:]

def build_input_and_labels(tokens, device):
    tokens = tokens.to(device)
    targets = torch.zeros(tokens.shape[0], dtype=int, device=device)
    batch_size = tokens.shape[0]

    for i in range(batch_size):
        targets[i] = torch.arange(tokens.shape[1])

    return tokens, targets

def prepare_data(BATCH_SIZE, N_BATCHES, TRAIN_RATIO):
    OUTPUT_DIR.mkdir()
    data_generator = make_data_generator(BATCH_SIZE)
    ds_tensor = create_dataset(data_generator, N_BATCHES)
    train_tensor, test_tensor = split_train_test(ds_tensor, TRAIN_RATIO)
    train_tokens, train_targets = build_input_and_labels(train_tensor, device)
    test_tokens, test_targets = build_input_and_labels(test_tensor, device)

    train_dataset = CustomDataset(train_tokens, train_targets)
    test_dataset = CustomDataset(test_tokens, test_targets)
    torch.save(train_dataset, OUTPUT_DIR / 'train_dataset.pt')
    torch.save(test_dataset, OUTPUT_DIR / 'test_dataset.pt')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    return train_loader, test_loader

 prepare_data()