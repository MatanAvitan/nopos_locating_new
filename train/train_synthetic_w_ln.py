import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from transformer_lens import HookedTransformer, HookedTransformerConfig, utils
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from utils import device
import os
from datetime import datetime
t=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

BASE = Path('.').resolve() 
N_CTX = 64
D_VOCAB = 64

TRAIN_RATIO = 0.8
N_BATCHES = 5_000
BATCH_SIZE = 8192
TBLOGSDIR = f'tblogs'

################ Data
class CustomDataset(TensorDataset):
    def __init__(self, tokens, targets):
        super().__init__(tokens, targets)

input_dir = Path('/home/nlp/matan_avitan/git/nopos_locating/datasets/abs_pos_pred_random_values')
train_dataset = torch.load(input_dir / 'train_dataset.pt', map_location=torch.device(device))
test_dataset = torch.load(input_dir / 'test_dataset.pt', map_location=torch.device(device))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
################ Data

cfg = HookedTransformerConfig(
    n_layers=3,
    d_model=128,
    d_head=128,
    n_heads=1,
    d_mlp=512,
    d_vocab=D_VOCAB,
    n_ctx=N_CTX,
    act_fn='relu',
    normalization_type='LN',
    device=device
)

def deactivate_position(model):
    model.pos_embed.W_pos.data[:] = 0.0
    model.pos_embed.W_pos.requires_grad = False


class LitTransformer(pl.LightningModule):
    def __init__(self, config, train_dataloader, val_dataloader):
        super().__init__()
        self.model = HookedTransformer(config)
        self.model.to(device)
        deactivate_position(self.model)
        print(self.model.pos_embed.W_pos.data)
        print(self.model.W_E.device)
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

    def forward(self, tokens):
        return self.model(tokens.to(device))

    def training_step(self, batch, batch_idx):
        tokens, targets = batch
        tokens = tokens.to(device)
        targets = targets .to(device)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
        scheduler = StepLR(optimizer, step_size=75, gamma=0.1)
        return [optimizer], [scheduler]

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
write_path.mkdir()
with open(write_path/'cfg', 'w') as f:
   f.write(str(cfg)) 
checkpoint_callback = ModelCheckpoint(dirpath=write_path, save_top_k=1, monitor='val_loss')
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = Trainer(max_epochs=250, accelerator='gpu', devices=1, logger=TensorBoardLogger('tblogs/'), callbacks=[checkpoint_callback, lr_monitor])

# Train the model
trainer.fit(lit_model)