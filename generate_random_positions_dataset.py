import torch
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from utils import device

torch.manual_seed(0)

class CustomDataset(TensorDataset):
    def __init__(self, tokens, targets):
        super().__init__(tokens, targets)

def make_data_generator(batch_size, d_vocab, ctx):
    while True:
        samples = np.random.randint(d_vocab, size=(batch_size, ctx))
        yield torch.tensor(samples, dtype=torch.long)

def create_dataset(data_generator, n_batches):
    unique_samples = set()
    for _ in range(n_batches):
        batch = next(data_generator)
        unique_samples.update(map(tuple, batch.tolist()))

    samples_tensor = torch.stack([torch.tensor(sample) for sample in unique_samples])
    return samples_tensor

def split_train_test(dataset, train_ratio):
    train_size = int(len(dataset) * train_ratio)
    return dataset[:train_size], dataset[train_size:]

def build_input_and_labels(tokens, device):
    targets = torch.arange(tokens.shape[1], device=device).expand_as(tokens)
    return tokens.to(device), targets

def prepare_data(batch_size, n_batches, train_ratio, d_vocab=64, ctx=64):
    output_dir = Path('datasets/abs_pos_pred_random_values')
    output_dir.mkdir(exist_ok=True)

    data_generator = make_data_generator(batch_size, d_vocab, ctx)
    dataset = create_dataset(data_generator, n_batches)
    train_tensor, test_tensor = split_train_test(dataset, train_ratio)

    train_tokens, train_targets = build_input_and_labels(train_tensor, device)
    test_tokens, test_targets = build_input_and_labels(test_tensor, device)

    train_dataset = CustomDataset(train_tokens, train_targets)
    test_dataset = CustomDataset(test_tokens, test_targets)
    
    torch.save(train_dataset, output_dir / 'train_dataset.pt')
    torch.save(test_dataset, output_dir / 'test_dataset.pt')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

# Example usage
cfg = {
    "batch_size": 512,
    "n_batches": 1000,
    "train_ratio": 0.8
}
train_loader, test_loader = prepare_data(**cfg)
