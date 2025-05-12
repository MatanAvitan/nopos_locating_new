import os
import joblib
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformerConfig
from utils import tokenize_fn
from nopos_lit_model import NoposLitTransformer
from torch.optim.lr_scheduler import OneCycleLR

# ─── Config ───────────────────────────────────────────────────────────────────
BASE       = Path('.').resolve()
TBLOGSDIR  = '/home/nlp/matan_avitan/tblogs'
N_CTX      = 64
D_MODEL    = 2_048
MLP_HIDDEN = 4*D_MODEL
BATCH_SIZE = 1_024 
EPOCHS     = 1_000
BASE_LR    = 1e-3
WEIGHT_DECAY = 1e-2
TRAIN_AMOUNT_OF_SAMPLES = None 
TEST_AMOUNT_OF_SAMPLES  = 1_024
hf_cache_dir = '/home/nlp/matan_avitan/cache_dir'
device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── TensorBoard & HParams ───────────────────────────────────────────────────
script_name = Path(__file__).stem  # Get the name of the current script without the extension
run_name    = f"{script_name}_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
log_dir   = BASE / TBLOGSDIR / run_name
writer    = SummaryWriter(log_dir=log_dir)

# ─── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
D_VOCAB = tokenizer.vocab_size

# ─── Dataset Helpers ──────────────────────────────────────────────────────────
def get_tokens(dataset, tokenizer, n_samples, max_length=N_CTX):
    tokenize_fn_partial = lambda batch: tokenize_fn(batch, tokenizer=tokenizer, max_length=max_length)
    tokens = dataset.map(tokenize_fn_partial, batched=True, input_columns=["abstract"], remove_columns=["abstract"])
    return torch.tensor(tokens['input_ids'][:n_samples], dtype=torch.long)

def get_labels(n_samples, max_length=N_CTX):
    base = torch.arange(max_length)
    return base.unsqueeze(0).repeat(n_samples, 1)

# ─── Load & Tokenize ──────────────────────────────────────────────────────────
train_ds = load_dataset('ccdv/arxiv-summarization', split='train', cache_dir=hf_cache_dir)
test_ds  = load_dataset('ccdv/arxiv-summarization', split='test',  cache_dir=hf_cache_dir)\
           .select(range(TEST_AMOUNT_OF_SAMPLES))
TRAIN_AMOUNT_OF_SAMPLES=len(train_ds)
train_tokens = get_tokens(train_ds, tokenizer, TRAIN_AMOUNT_OF_SAMPLES)
test_tokens  = get_tokens(test_ds,  tokenizer, TEST_AMOUNT_OF_SAMPLES)

# ─── Print Top k Most Popular Tokens ──────────────────────────────────────────
def write_top_k_tokens(tokens, tokenizer, k=10):
    # Flatten the tokens tensor to count all tokens
    flattened_tokens = tokens.view(-1)
    # Count occurrences of each token
    token_counts = torch.bincount(flattened_tokens)
    # Get the indices of the top k most frequent tokens
    top_k_indices = list(torch.topk(token_counts, k).indices)
    joblib.dump(top_k_indices, 'top_k_indices.pkl')


# Call the function
write_top_k_tokens(train_tokens, tokenizer, k=4*2_048)