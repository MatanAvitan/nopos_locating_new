import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from evaluation import evaluate
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from utils import device
import os
from datetime import datetime
import numpy as np
import random
import json
# Set random seeds for reproducibility
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
t=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# Extract the filename without extension to use as the experiment name
experiment_name = os.path.splitext(os.path.basename(__file__))[0]
# Add a timestamp to the experiment name
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_name_with_timestamp = f"{experiment_name}_{timestamp}_seed={seed}"

BASE = Path('.').resolve() 
N_CTX = 64
D_VOCAB = 5_000
EPOCHS = 400
TRAIN_RATIO = 0.8
N_BATCHES = 100
# BATCH_SIZE = 2048
BATCH_SIZE = 8_192
TBLOGSDIR = f'tblogs'

################ Data
class CustomDataset(TensorDataset):
    def __init__(self, tokens, targets):
        super().__init__(tokens, targets)

# input_dir = Path('/home/nlp/matan_avitan/git/nopos_locating/datasets/abs_pos_pred_random_values')
# train_dataset = torch.load(input_dir / 'train_dataset.pt', map_location=torch.device(device))
# test_dataset = torch.load(input_dir / 'test_dataset.pt', map_location=torch.device(device))
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

train_dataset = torch.randint(low=0, high=D_VOCAB, size=(BATCH_SIZE*N_BATCHES, N_CTX))
test_dataset = torch.randint(low=0, high=D_VOCAB, size=(BATCH_SIZE*N_BATCHES, N_CTX))

# Create labels as a sequence of 0 to N_CTX
labels = torch.arange(N_CTX).expand(train_dataset.size(0), -1)

# Update the train and test datasets
train_dataset = TensorDataset(train_dataset, labels)
test_dataset = TensorDataset(test_dataset, labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
################ Data

cfg = HookedTransformerConfig(
    n_layers=1,
    d_model=1024,
    d_head=1024,
    n_heads=1,
    d_mlp=4096,
    d_vocab=D_VOCAB,
    n_ctx=N_CTX,
    act_fn='relu',
    normalization_type='LN',
    device=device
)

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

def freeze_lns(model, l=0):
    model.blocks[0].ln1.w.data[:] = 1
    model.blocks[0].ln1.w.requires_grad = False
    model.blocks[0].ln1.b.data[:] = 0
    model.blocks[0].ln1.b.requires_grad = False

    model.blocks[0].ln2.w.data[:] = 1
    model.blocks[0].ln2.w.requires_grad = False
    model.blocks[0].ln2.b.data[:] = 0
    model.blocks[0].ln2.b.requires_grad = False

class LitTransformer(pl.LightningModule):
    def __init__(self, config, train_dataloader, val_dataloader):
        super().__init__()
        self.model = HookedTransformer(config)
        self.model.to(device)
        deactivate_position(self.model)
        freeze_embeddings(self.model)
        freeze_attention(self.model, l=0)
        freeze_lns(self.model, l=0)
        print(self.model.pos_embed.W_pos.data)
        print(self.model.blocks[0].attn.W_Q.data)
        print(self.model.W_E.device)
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

    def forward(self, tokens):
        return self.model(tokens.to(device))

    def training_step(self, batch, batch_idx):
        tokens, targets = batch
        tokens = tokens.to(device)
        targets = targets.to(device)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    # def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.95))
        # scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
        # return [optimizer], [scheduler]

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

# Initialize model
lit_model = LitTransformer(cfg, train_loader, test_loader)
lit_model.to(device)

# Setup the trainer
write_path = Path(f'models/synthetic_abs_pos_{t}')
write_path.mkdir(parents=True, exist_ok=True)
with open(write_path/'cfg', 'w') as f:
   f.write(str(cfg)) 
checkpoint_callback = ModelCheckpoint(dirpath=write_path, save_top_k=2, monitor='val_loss')
lr_monitor = LearningRateMonitor(logging_interval='step')
# Create the TensorBoard logger with the experiment name
logger = TensorBoardLogger('tblogs/', name=experiment_name_with_timestamp)
trainer = Trainer(max_epochs=EPOCHS, 
                  accelerator='gpu',
                  devices=1, logger=logger, callbacks=[checkpoint_callback, lr_monitor],
                  accumulate_grad_batches=2  # Simulates doubling batch size
                  )
# Train the model
trainer.fit(lit_model)

results = evaluate(lit_model.model, test_loader, device)
with open(write_path/'results', 'w') as f:
   f.write(str({'results_for_last_model': results}))

# Save results as JSON for easier consolidation
results_dict = {
    'experiment_name': experiment_name,
    'timestamp': timestamp,
    'accuracy': float(results[0]) if isinstance(results, tuple) else float(results.get('accuracy', 0)),
    'loss': float(results[1]) if isinstance(results, tuple) and len(results) > 1 else float(results.get('loss', 0)),
    'hyperparameters': {
        'n_ctx': N_CTX,
        'd_vocab': D_VOCAB,
        'd_model': cfg.d_model,
        'd_mlp': cfg.d_mlp,
        'n_layers': cfg.n_layers,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'n_batches': N_BATCHES,
        'seed': seed
    }
}
with open(write_path/'results.json', 'w') as f:
   json.dump(results_dict, f, indent=2)
print(f"Results: Accuracy={results_dict['accuracy']:.4f}, Loss={results_dict['loss']:.4f}") 
