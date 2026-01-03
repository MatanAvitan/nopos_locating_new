import os
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

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
from transformers import AutoTokenizer

# ─── Config ───────────────────────────────────────────────────────────────────
IS_FIRST = True
MODEL = 'mistral_v1'
BASE       = Path('.').resolve()
TBLOGSDIR  = '/home/nlp/matan_avitan/tblogs'
N_CTX      = 64
D_MODEL    = 2_048
MLP_HIDDEN = 4*D_MODEL
BATCH_SIZE = 2_048
EPOCHS     = 10_000
BASE_LR    = 5e-4
WEIGHT_DECAY = 1e-1
TRAIN_AMOUNT_OF_SAMPLES = None 
TEST_AMOUNT_OF_SAMPLES  = 1_024
hf_cache_dir = '/home/nlp/matan_avitan/cache_dir'
device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── TensorBoard & HParams ───────────────────────────────────────────────────
script_name = Path(__file__).stem  # Get the name of the current script without the extension
run_name    = f"{script_name}_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
log_dir   = BASE / TBLOGSDIR / run_name
writer    = SummaryWriter(log_dir=log_dir)

hparams = {
    'n_ctx':          N_CTX,
    'd_model':        D_MODEL,
    'd_mlp':          MLP_HIDDEN,
    'batch_size':     BATCH_SIZE,
    'epochs':         EPOCHS,
    'base_lr':        BASE_LR,
    'weight_decay':   WEIGHT_DECAY,
    'train_samples':  TRAIN_AMOUNT_OF_SAMPLES,
    'test_samples':   TEST_AMOUNT_OF_SAMPLES,
}

# ─── Prepare best‐model tracking ───────────────────────────────────────────────
best_val_loss        = float('inf')
best_val_acc         = 0.0
best_checkpoint_path = log_dir / "best_mlp.pt"

# ─── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer =  AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
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
model     = lit_model.model

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

train_tokens_loader = DataLoader(TensorDataset(train_tokens), batch_size=BATCH_SIZE,
                                 shuffle=True,  pin_memory=True, num_workers=64,
                                 prefetch_factor=4,            # load 4 batches ahead
                                 )
test_tokens_loader  = DataLoader(TensorDataset(test_tokens),  batch_size=BATCH_SIZE,
                                 shuffle=False, pin_memory=True, num_workers=64,
                                 prefetch_factor=4,            # load 4 batches ahead
                                 )

# ─── Precompute Embeddings ─────────────────────────────────────────────────────
layers_to_cache = ['blocks.0.ln2.hook_normalized']
def precompute_embeddings(dl, model):
    embs = []
    for (x,) in tqdm(dl, desc="Precompute"):
        _, cache = model.run_with_cache(x.to(device), names_filter=layers_to_cache)
        acts = cache[layers_to_cache[0]].detach().cpu()
        embs.append(acts)
    return torch.cat(embs, dim=0)
if IS_FIRST:
    train_embeddings = precompute_embeddings(train_tokens_loader, model)
    test_embeddings  = precompute_embeddings(test_tokens_loader,  model)
    torch.save(train_embeddings, f'/home/nlp/matan_avitan/ln_rep_prediction/train_embeddings_tokenizer_{MODEL}_D_MODEL_{D_MODEL}.pt')
    torch.save(test_embeddings,  f'/home/nlp/matan_avitan/ln_rep_prediction/test_embeddings_tokenizer_{MODEL}_D_MODEL_{D_MODEL}.pt')
else:
    train_embeddings = torch.load(f'/home/nlp/matan_avitan/ln_rep_prediction/train_embeddings_tokenizer_{MODEL}_D_MODEL_{D_MODEL}.pt')
    test_embeddings  = torch.load(f'/home/nlp/matan_avitan/ln_rep_prediction/test_embeddings_tokenizer_{MODEL}_D_MODEL_{D_MODEL}.pt')
train_labels     = get_labels(TRAIN_AMOUNT_OF_SAMPLES, N_CTX)
test_labels      = get_labels(TEST_AMOUNT_OF_SAMPLES,  N_CTX)

train_loader = DataLoader(TensorDataset(train_embeddings, train_labels),
                          batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True, num_workers=16)
test_loader  = DataLoader(TensorDataset(test_embeddings,  test_labels),
                          batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=16)

# ─── Compute Average Embedding Vector ─────────────────────────────────────────
# with torch.no_grad():
    # vocab_embeddings = (model.W_E.data @ (model.W_V.data.squeeze(0).squeeze(0) @ model.W_O.data.squeeze(0).squeeze(0)))
    # avg_embedding = vocab_embeddings.mean(dim=0)  # Shape: (d_model,)
 
# ─── MLP Definition ────────────────────────────────────────────────────────────
class PositionPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # # Initialize the first layer weights
        # with torch.no_grad():
        #     self.mlp[0].weight.data = torch.normal(mean=0.0, std=0.1, size=self.mlp[0].weight.data.size(), device=device)  # Gaussian noise
        #     self.mlp[0].weight.data += avg_embedding.unsqueeze(0)  # Broadcast avg_embedding
        #     self.mlp[0].bias.data.zero_()  # Set bias to zero

    def forward(self, x):
        return self.mlp(x)

mlp_model = PositionPredictorMLP(D_MODEL, MLP_HIDDEN, N_CTX).to(device)
#    - use reduce-overhead mode if you hit Dynamo errors
#    - pass a tiny example_input so shapes become static
# mlp_model = torch.compile(
#     mlp_model,
#     backend="inductor",
#     mode="reduce-overhead",
#     fullgraph=True
# )
# # 3) warm up the graph
# _ = mlp_model(torch.randn(1, D_MODEL, device=device))

if device == "cuda" and torch.cuda.device_count() > 1:
    mlp_model = nn.DataParallel(mlp_model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(mlp_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
scheduler = OneCycleLR(
    optimizer,
    max_lr=BASE_LR * 10,
    total_steps=EPOCHS * len(train_loader),
    pct_start=0.1,
    anneal_strategy="cos"
)
scaler = torch.cuda.amp.GradScaler()

# ─── Training Loop ─────────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    # — Training —
    mlp_model.train()
    running_loss = correct = total = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        logits = mlp_model(X)
        loss   = criterion(logits, y)

        optimizer.zero_grad()
        # 1) scale the loss
        scaler.scale(loss).backward()
        # 2) step the optimizer through the scaler
        scaler.step(optimizer)
        # 3) update the scaler for the next iteration
        scaler.update()
        # 4) step your LR scheduler after the optimizer (if you use one)
        scheduler.step()

        running_loss += loss.item() * X.size(0)
        preds        = logits.argmax(dim=1)
        correct     += (preds == y).sum().item()
        total       += y.size(0)

    train_loss = running_loss / total
    train_acc  = correct      / total

    # — Validation —
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

    # — Checkpoint if improved —
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     state_dict = (mlp_model.module.state_dict()
    #                   if isinstance(mlp_model, nn.DataParallel)
    #                   else mlp_model.state_dict())
    #     torch.save(state_dict, best_checkpoint_path)
    #     print(f"→ Saved new best model at epoch {epoch} (val_loss={val_loss:.4f})")

    # — TensorBoard logs —
    writer.add_scalar('Loss/train',     train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc,  epoch)
    writer.add_scalar('Loss/val',       val_loss,   epoch)
    writer.add_scalar('Accuracy/val',   val_acc,    epoch)
    writer.add_scalar('LearningRate',   optimizer.param_groups[0]['lr'], epoch)

    print(f"Epoch {epoch}/{EPOCHS} "
          f"Train ({train_loss:.4f}, acc={train_acc:.4f})  "
          f"Val   ({val_loss:.4f}, acc={val_acc:.4f})")

# ─── Final HParams record ─────────────────────────────────────────────────────
metrics = {
    'hparam/val_loss': best_val_loss,
    'hparam/val_acc':  best_val_acc
}
writer.add_hparams(hparams, metrics)

writer.close()

