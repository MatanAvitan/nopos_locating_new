import torch
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from utils import device

torch.manual_seed(0)


###########
import os
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]

# Paths to the dataset files
BASE = '/home/nlp/matan_avitan/git/nopos_locating/datasets/wikitext-103/enwik8'
train_file = f'{BASE}/train.txt.raw'
valid_file = f'{BASE}/valid.txt.raw'
test_file = f'{BASE}/test.txt.raw'

# Load datasets
train_dataset = TextDataset(train_file)
valid_dataset = TextDataset(valid_file)
test_dataset = TextDataset(test_file)

# DataLoader parameters
batch_size = 512  # Adjust batch size as necessary
shuffle = True  # Typically, you want to shuffle the training data

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example usage
for batch in train_loader:
    print(batch)  # This will print out batches of text characters
    break

###########
