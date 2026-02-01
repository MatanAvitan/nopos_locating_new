import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Literal

# Configuration
checkpoint_path = "nanoGPT/out-2layer-mechanism-bos80/R0/final_ckpt.pt"
device = "cpu"
n_samples = 100
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
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask[:,:,:T,:T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
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
        x = self.block2(x)
        x = self.ln_f(x)
        return x

def analyze():
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
    model = CustomModel(gpt_conf)
    state_dict = checkpoint['model']
    model_state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    tokens = torch.randint(0, gpt_conf.vocab_size, (n_samples, gpt_conf.block_size))
    with torch.no_grad():
        x = model.wte(tokens)
        x = model.block1(x)
        layer = model.block2
        x_ln = layer.ln_1(x)
        
        n_head = gpt_conf.n_head
        head_dim = gpt_conf.n_embd // n_head
        W_Q, W_K, W_V = layer.attn.c_attn.weight.split(gpt_conf.n_embd, dim=0)
        b_Q, b_K, b_V = layer.attn.c_attn.bias.split(gpt_conf.n_embd, dim=0)
        
        K = torch.matmul(x_ln, W_K.T) + b_K
        Q = torch.matmul(x_ln, W_Q.T) + b_Q
        K = K.view(n_samples, gpt_conf.block_size, n_head, head_dim).transpose(1, 2)
        Q = Q.view(n_samples, gpt_conf.block_size, n_head, head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        mask = torch.tril(torch.ones(gpt_conf.block_size, gpt_conf.block_size)).view(1, 1, gpt_conf.block_size, gpt_conf.block_size)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        probs = F.softmax(scores, dim=-1)
        
        target_probs = probs[:, :, bos_pos+1:, 0].mean(dim=(0, 2))
        pos0_head = torch.argmax(target_probs).item()
        
        K_pos0 = K[:, :, 0, :]
        variances = K_pos0.var(dim=0).mean(dim=-1)
        mean_K_pos0 = K_pos0.mean(dim=0)
        mean_norms = torch.norm(mean_K_pos0, dim=-1)
        
        print(f"Pos 0 head: {pos0_head}")
        print(f"Pos 0 head variance: {variances[pos0_head].item():.6f}")
        print(f"Pos 0 head mean norm: {mean_norms[pos0_head].item():.6f}")
        
        other_heads = [h for h in range(n_head) if h != pos0_head]
        avg_other_var = variances[other_heads].mean().item()
        avg_other_norm = mean_norms[other_heads].mean().item()
        
        print(f"Avg other heads variance: {avg_other_var:.6f}")
        print(f"Avg other heads mean norm: {avg_other_norm:.6f}")
        
        K_pos80 = K[:, pos0_head, bos_pos, :]
        var_pos80 = K_pos80.var(dim=0).mean().item()
        norm_pos80 = torch.norm(K_pos80.mean(dim=0)).item()
        print(f"Pos 80 (BOS) variance in pos0_head: {var_pos80:.6f}")
        print(f"Pos 80 (BOS) mean norm in pos0_head: {norm_pos80:.6f}")

if __name__ == "__main__":
    analyze()
