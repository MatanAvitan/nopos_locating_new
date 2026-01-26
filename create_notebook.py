import nbformat as nbf

nb = nbf.v4.new_notebook()

code = """import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Literal

# Configuration
checkpoint_path = "nanoGPT/out-2layer-mechanism-bos80/R0/final_ckpt.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
pos0_head_idx = 7
bos_pos = 80

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    use_positional_embedding: bool = False
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    log_attention_stats: bool = False
    skip_ln2: bool = False

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        self.last_q = q
        self.last_k = k
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask[:,:,:T,:T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        self.last_attn = att # Store for visualization
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.block1 = Block(config)
        self.block2 = Block(config)
        self.ln_f = LayerNorm(config.n_embd, config.bias)
    def forward(self, idx):
        x = self.wte(idx)
        x = self.block1(x)
        # Capture input to block2's attention (post LN1)
        self.block2_input_ln1 = self.block2.ln_1(x)
        x = self.block2(x)
        x = self.ln_f(x)
        return x

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
config_dict = checkpoint['config']
gpt_conf = GPTConfig(
    n_layer=2,
    n_head=config_dict['n_head'],
    n_embd=config_dict['n_embd'],
    block_size=config_dict['block_size'],
    vocab_size=config_dict['vocab_size'],
    bias=config_dict.get('bias', True)
)

# Initialize and load model
model = CustomModel(gpt_conf)
state_dict = checkpoint['model']
model_state_dict = model.state_dict()
new_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()

# Extract weights for head 7
layer = model.block2
n_head = gpt_conf.n_head
head_dim = gpt_conf.n_embd // n_head
c_attn_weight = layer.attn.c_attn.weight
c_attn_bias = layer.attn.c_attn.bias
W_Q_all, W_K_all, W_V_all = c_attn_weight.split(gpt_conf.n_embd, dim=0)
b_Q_all, b_K_all, b_V_all = c_attn_bias.split(gpt_conf.n_embd, dim=0)
W_Q = W_Q_all.view(n_head, head_dim, gpt_conf.n_embd)[pos0_head_idx]
W_K = W_K_all.view(n_head, head_dim, gpt_conf.n_embd)[pos0_head_idx]
b_Q = b_Q_all.view(n_head, head_dim)[pos0_head_idx]
b_K = b_K_all.view(n_head, head_dim)[pos0_head_idx]

# Perform forward pass
tokens = torch.randint(0, gpt_conf.vocab_size, (1, gpt_conf.block_size), device=device)
with torch.no_grad():
    _ = model(tokens)

# Get activations
x_i = model.block2_input_ln1[0] # [T, n_embd]
q_i = model.block2.attn.last_q[0, pos0_head_idx] # [T, head_dim]
k_i = model.block2.attn.last_k[0, pos0_head_idx] # [T, head_dim]

print(f"Model loaded. Variables available:")
print(f" - x_i: Input to block2 attention (post-LN1) [T, n_embd]")
print(f" - q_i: Queries for Head {pos0_head_idx} [T, head_dim]")
print(f" - k_i: Keys for Head {pos0_head_idx} [T, head_dim]")
print(f" - W_Q, W_K, b_Q, b_K: Weights/Biases for Head {pos0_head_idx}")

# Validation Plot: Attention
attn_weights = model.block2.attn.last_attn[0, pos0_head_idx].cpu().numpy()
plt.figure(figsize=(8, 6))
plt.imshow(attn_weights, cmap='viridis')
plt.title(f'Block 2 Head {pos0_head_idx} Attention Map')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.axvline(x=0, color='r', linestyle='--', label='Pos 0')
plt.axhline(y=80, color='w', linestyle=':', label='BOS Pos')
plt.colorbar()
plt.show()

# Dot Product Investigation
scores = (q_i @ k_i.T) / math.sqrt(head_dim)
scores = scores.cpu().numpy()

plt.figure(figsize=(10, 4))
plt.plot(scores[100, :], label='Scores from Q at pos 100')
plt.axvline(x=0, color='r', linestyle='--', label='Pos 0')
plt.axvline(x=80, color='g', linestyle='-.', label='BOS (Pos 80)')
plt.title(f'Attention Scores for Query at Pos 100 (Head {pos0_head_idx})')
plt.xlabel('Key Position')
plt.ylabel('Score (pre-softmax)')
plt.legend()
plt.show()

print(f"Score at Pos 0: {scores[100, 0]:.2f}")
print(f"Score at Pos 80: {scores[100, 80]:.2f}")
print(f"Mean score (other pos): {np.mean(scores[100, 1:80]):.2f}")
"""

nb['cells'] = [nbf.v4.new_code_cell(code)]

with open('investigate_bos80_head7.ipynb', 'w') as f:
    nbf.write(nb, f)
