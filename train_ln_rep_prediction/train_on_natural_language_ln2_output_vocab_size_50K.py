import os
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

# ─── Config ───────────────────────────────────────────────────────────────────
BASE       = Path('.').resolve()
TBLOGSDIR  = '/dccstor/ai_security2/matan/tblogs'
N_CTX      = 64
D_MODEL    = 2_048
BATCH_SIZE = 64
EPOCHS     = 1_000
BASE_LR    = 1e-3
WEIGHT_DECAY = 1e-2
TRAIN_AMOUNT_OF_SAMPLES = GPT2TokenizerFast.from_pretrained('gpt2').vocab_size
TEST_AMOUNT_OF_SAMPLES  = 1_024
hf_cache_dir = '/dccstor/ai_security2/matan/cache_dir'
device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── TensorBoard ──────────────────────────────────────────────────────────────
run_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
log_dir = Path(TBLOGSDIR) / run_name
# log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# ─── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
D_VOCAB = tokenizer.vocab_size

# ─── Model with Hooks ──────────────────────────────────────────────────────────
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
lit_model = NoposLitTransformer(cfg, None, None).to(device)
model = lit_model.model

# ─── Dataset Helpers ──────────────────────────────────────────────────────────
def get_tokens(dataset, tokenizer, n_samples, max_length=N_CTX):
    tokenize_fn_partial = lambda batch: tokenize_fn(batch, tokenizer=tokenizer, max_length=max_length)
    tokens = dataset.map(tokenize_fn_partial, batched=True, input_columns=["abstract"], remove_columns=["abstract"])
    return torch.tensor(tokens['input_ids'][:n_samples], dtype=torch.long)

def get_labels(n_samples, max_length=N_CTX):
    base = torch.arange(max_length)
    return base.unsqueeze(0).repeat(n_samples, 1)

# ─── Load & Tokenize ──────────────────────────────────────────────────────────
train_ds = load_dataset('ccdv/arxiv-summarization', split='train', cache_dir=hf_cache_dir).select(range(TRAIN_AMOUNT_OF_SAMPLES))
test_ds  = load_dataset('ccdv/arxiv-summarization', split='test',  cache_dir=hf_cache_dir).select(range(TEST_AMOUNT_OF_SAMPLES))

train_tokens = get_tokens(train_ds, tokenizer, TRAIN_AMOUNT_OF_SAMPLES)
test_tokens  = get_tokens(test_ds,  tokenizer, TEST_AMOUNT_OF_SAMPLES)

train_tokens_loader = DataLoader(TensorDataset(train_tokens), batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True, num_workers=16)
test_tokens_loader  = DataLoader(TensorDataset(test_tokens),  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=16)

# ─── Precompute Embeddings ─────────────────────────────────────────────────────
layers_to_cache = ['blocks.0.ln2.hook_normalized']
def precompute_embeddings(dl, model):
    embs = []
    for (x,) in tqdm(dl, desc="Precompute"):
        _, cache = model.run_with_cache(x.to(device), names_filter=layers_to_cache)
        acts = cache[layers_to_cache[0]].detach().cpu()
        embs.append(acts)
    return torch.cat(embs, dim=0)

train_embeddings = precompute_embeddings(train_tokens_loader, model)
test_embeddings  = precompute_embeddings(test_tokens_loader,  model)
train_labels     = get_labels(TRAIN_AMOUNT_OF_SAMPLES, N_CTX)
test_labels      = get_labels(TEST_AMOUNT_OF_SAMPLES,  N_CTX)

train_loader = DataLoader(TensorDataset(train_embeddings, train_labels), batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True, num_workers=16)
test_loader  = DataLoader(TensorDataset(test_embeddings,  test_labels),  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=16)

# ─── MLP Definition ────────────────────────────────────────────────────────────
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

mlp_model = PositionPredictorMLP(input_dim=D_MODEL, hidden_dim=4*D_MODEL, output_dim=N_CTX).to(device)
criterion  = nn.CrossEntropyLoss()
optimizer  = optim.AdamW(mlp_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

# ─── Training Loop ─────────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    # Training
    mlp_model.train()
    running_loss = correct = total = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        logits = mlp_model(X)
        loss   = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct   += (preds == y).sum().item()
        total     += y.size(0)

    train_loss = running_loss / total
    train_acc  = correct      / total

    # Validation
    mlp_model.eval()
    val_loss = val_correct = val_total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = mlp_model(X)
            loss   = criterion(logits, y)
            val_loss    += loss.item() * X.size(0)
            preds       = logits.argmax(dim=1)
            val_correct += (preds == y).sum().item()
            val_total   += y.size(0)

    val_loss = val_loss / val_total
    val_acc  = val_correct / val_total

    # ─── TensorBoard Logging ────────────────────────────────────────────────────
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Loss/val',   val_loss,   epoch)
    writer.add_scalar('Accuracy/val', val_acc,   epoch)
    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

    print(f"Epoch {epoch}/{EPOCHS}  "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
          f"Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}")

writer.close()

