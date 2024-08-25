import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

class TextDataset(Dataset):
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
        
        # Load and filter the data
        with open(file_path, 'r') as file:
            self.data = [
                line.strip() for line in file.readlines()
                if len(self.tokenizer.encode(line.strip(), add_special_tokens=False)) >= self.max_length
            ]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the line at the specified index
        line = self.data[idx]
        
        # Tokenize the line using the GPT2 tokenizer
        tokens = self.tokenizer.encode(
            line,
            add_special_tokens=False,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # The tokenizer returns a tensor with shape [1, max_length], so we need to squeeze it
        tokens = tokens.squeeze(0)
        
        return tokens

# Example usage:
BASE = '/home/nlp/matan_avitan/git/nopos_locating/datasets/wikitext-103/enwik8'
train_file = f'{BASE}/train.txt.raw'
valid_file = f'{BASE}/valid.txt.raw'
test_file = f'{BASE}/test.txt.raw'

# Initialize the GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load datasets
train_dataset = TextDataset(train_file, tokenizer)
valid_dataset = TextDataset(valid_file, tokenizer)
test_dataset = TextDataset(test_file, tokenizer)

# DataLoader parameters
batch_size = 1024  # Adjust batch size as necessary

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example usage
for batch in train_loader:
    print(batch)
    break