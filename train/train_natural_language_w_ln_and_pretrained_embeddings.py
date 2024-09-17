from transformers import GPT2Model
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from evaluation import evaluate

from transformer_lens import HookedTransformer, HookedTransformerConfig, utils
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from utils import device
import os
from datetime import datetime
t=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

BASE = Path('.').resolve() 
N_CTX = 128 
D_VOCAB = 50_257

BATCH_SIZE = 256
TBLOGSDIR = f'tblogs'

################ Data
class CustomDataset(TensorDataset):
    def __init__(self, tokens, targets):
        super().__init__(tokens, targets)

input_dir = Path('/home/nlp/matan_avitan/git/nopos_locating/datasets/pos_pred_natural_language')
train_dataset = torch.load(input_dir / 'train_dataset.pt', map_location=torch.device(device))
test_dataset = torch.load(input_dir / 'test_dataset.pt', map_location=torch.device(device))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print(f':={train_dataset.shape}')
print(f':={test_dataset.shape}')
################ Data

cfg = HookedTransformerConfig(
    n_layers=6,
    d_model=768,
    d_head=32,
    n_heads=1,
    d_mlp=128,
    d_vocab=D_VOCAB,
    n_ctx=N_CTX,
    act_fn='relu',
    normalization_type="LN",
    device=device,
    use_hook_mlp_in=True
)

def deactivate_position(model):
    model.pos_embed.W_pos.data[:] = 0.0
    model.pos_embed.W_pos.requires_grad = False

def get_gpt2_word_embeddings():
    # Load the GPT-2 model
    model = GPT2Model.from_pretrained('gpt2')
    # Get the word embeddings matrix (first layer of the network)
    word_embeddings = model.get_input_embeddings()
    # Convert to a tensor
    embedding_matrix = word_embeddings.weight
    # If needed, you can convert it to a numpy array for easier manipulation
    # embedding_matrix = embedding_matrix.detach().numpy()
    print(embedding_matrix.shape)  # This will print the shape of the embedding matrix
    print(type(embedding_matrix))
    return embedding_matrix

def freeze_embeddings(model):
    model.embed.W_E.requires_grad = False

def set_embeddings(model):
    word_embeddings_tr = get_gpt2_word_embeddings()
    word_embeddings_tr.requires_grad = False
    model.embed.W_E = word_embeddings_tr

class LitTransformer(pl.LightningModule):
    def __init__(self, config, train_dataloader, val_dataloader):
        super().__init__()
        self.model = HookedTransformer(config)
        self.model.to(device)
        deactivate_position(self.model)
        set_embeddings(self.model)
        freeze_embeddings(self.model)
        print(self.model.pos_embed.W_pos.data)
        print(self.model.W_E.device)
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

    def forward(self, tokens):
        return self.model(tokens.to(device))

    def build_targets(self, tokens):
        batch_size, ctx = tokens.shape
        row_arange = torch.arange(ctx, device=device)
        targets = row_arange.unsqueeze(0).repeat(batch_size, 1)
        return targets

    def training_step(self, batch, batch_idx):
        print(f'Train batch size: {batch.shape}')
        tokens = batch
        tokens = tokens.to(device)
        targets = self.build_targets(tokens)
        targets = targets.to(device)
        logits = self(tokens)
        loss = self.loss_fn(logits, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        print(f'Validation batch size: {batch.shape}')
        tokens = batch
        tokens = tokens.to(device)
        targets = self.build_targets(tokens)
        targets = targets.to(device)
        logits = self(tokens)
        loss = self.loss_fn(logits, targets)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
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
run_name = f'natural_language_abs_pos_w_ln_w_gpt_word_embeddings{t}'
write_path = Path(f'models/{run_name}')
write_path.mkdir()
with open(write_path/'cfg', 'w') as f:
   f.write(str(cfg)) 
checkpoint_callback = ModelCheckpoint(dirpath=write_path, save_top_k=2, monitor='val_loss')
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = Trainer(max_epochs=4000, accelerator='gpu', devices=1, logger=TensorBoardLogger('tblogs/', name=run_name), callbacks=[checkpoint_callback, lr_monitor])

# Train the model
trainer.fit(lit_model)

results = evaluate(lit_model.model, test_loader, device)
with open(write_path/'results', 'w') as f:
   f.write(str({'results_for_last_model': results})) 
