import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from pathlib import Path

class TextDataset:
    def __init__(self, file_path, tokenizer, max_length=512):
        """
        Args:
            file_path (str): Path to the text file.
            tokenizer (transformers.PreTrainedTokenizer): Hugging Face GPT2 tokenizer.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load, filter, and tokenize the data
        self.data = self._load_and_filter_data()
        
    def _load_and_filter_data(self):
        # Filter and tokenize lines with at least 512 tokens
        tokenized_data = []
        with open(self.file_path, 'r') as file:
            for line in file:
                line = line.strip()
                tokens = self.tokenizer.encode(
                    line,
                    add_special_tokens=False,
                    max_length=self.max_length,
                    truncation=True
                )
                if len(tokens) >= self.max_length:
                    tokens_tensor = torch.tensor(tokens[:self.max_length], dtype=torch.long)
                    tokenized_data.append(tokens_tensor)
        
        # Stack all tokenized tensors into a single tensor
        return torch.stack(tokenized_data)

    def save_to_disk(self, output_path):
        # Save the tensor data to disk
        torch.save(self.data, output_path)

# Example usage:
BASE = '/home/nlp/matan_avitan/git/nopos_locating/datasets/wikitext-103/enwik8'
train_file = f'{BASE}/train.txt.raw'
valid_file = f'{BASE}/valid.txt.raw'
test_file = f'{BASE}/test.txt.raw'

# Paths to save the tokenized tensor data
output_dir = Path('/home/nlp/matan_avitan/git/nopos_locating/datasets/pos_pred_natural_language')
output_dir.mkdir(parents=True, exist_ok=True)
train_output_path = output_dir / 'train_dataset.pt'
valid_output_path = output_dir / 'valid_dataset.pt'
test_output_path = output_dir / 'test_dataset.pt'

# Initialize the GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load, filter, and save datasets
train_dataset = TextDataset(train_file, tokenizer)
train_dataset.save_to_disk(train_output_path)

valid_dataset = TextDataset(valid_file, tokenizer)
valid_dataset.save_to_disk(valid_output_path)

test_dataset = TextDataset(test_file, tokenizer)
test_dataset.save_to_disk(test_output_path)
