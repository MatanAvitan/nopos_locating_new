import matplotlib.pyplot as plt
import einops
from fancy_einsum import einsum
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
import math
import tqdm.auto as tqdm
import plotly.express as px
from transformer_lens.utils import to_numpy
import plotly.io as pio

# Set plotly to create pngs
## Set default template to 'plotly_white' for a clean academic style
pio.templates.default = "plotly_white"
## Customize the global layout for academic style
pio.templates["plotly_white"]['layout'].update({
    'font': {
        'family': 'Times New Roman',  # Common academic font
        'size': 18                    # Larger font for readability in papers
    },
    'margin': dict(l=50, r=50, t=50, b=50),  # Adjust margins for clearer plots
    'height': 400,  # Adjust plot size
    'width': 800    # Adjust plot size
});
pio.renderers.default = "png"

# Utils
def line(tensor, line_labels=None, yaxis="", xaxis="", **kwargs):
    tensor = to_numpy(tensor)
    labels = {"y": yaxis, "x": xaxis}
    fig = px.line(tensor, labels=labels, **kwargs)
    if line_labels:
        for c, label in enumerate(line_labels):
            fig.data[c].name = label
    fig.layout.height = 400
    fig.show()
    return fig
    

def imshow(tensor, yaxis="", xaxis="", **kwargs):
    tensor = to_numpy(tensor)
    plot_kwargs = {
        "color_continuous_scale": "RdBu",
        "color_continuous_midpoint": 0.0,
        "labels": {"x": xaxis, "y": yaxis},
    }
    plot_kwargs.update(kwargs)
    fig = px.imshow(tensor, **plot_kwargs)
    fig.show()
    return fig

##########################################################
#### Consts
##########################################################
mean, std = 0, None
SAMPLES = 5000
D_MODEL = 1_024
D_VOCAB = 256 
D_HEAD = 128
CTX = 256 
EPS = 1e-5
std = 0.8 / np.sqrt(D_MODEL)

# LayerNorm
class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(D_MODEL))
        self.b = nn.Parameter(torch.zeros(D_MODEL))

    def forward(self, residual):
        # residual: [batch, position, d_model]
        mean = residual.mean(dim=-1, keepdims=True)
        variance = residual.var(dim=-1, keepdims=True, unbiased=False)
        residual_norm = (residual - mean) / torch.sqrt(variance+EPS) # [batch, position, d_model]
        return einsum('batch position d_model, d_model -> batch position d_model', residual_norm, self.w) + self.b 

def apply_causal_mask(attn_scores):
    # attn_scores: [query_pos, key_pos]
     mask = torch.triu(torch.ones(*attn_scores.shape[-2:], dtype=bool, device=attn_scores.device), diagonal=1)
     return attn_scores.masked_fill_(mask, -np.inf)
print('start')
X = torch.normal(size=(SAMPLES, CTX, D_MODEL), mean=mean, std=std)
ln1 = LayerNorm()
normalized_input = ln1(X)
K_W = torch.normal(size=(D_MODEL, D_HEAD), mean=mean, std=std)
Q_W = torch.normal(size=(D_MODEL, D_HEAD), mean=mean, std=std)
V_W = torch.normal(size=(D_MODEL, D_HEAD), mean=mean, std=std)
K = einsum('samples ctx d_model, d_model d_head -> samples ctx d_head', normalized_input , K_W)
Q = einsum('samples ctx d_model, d_model d_head -> samples ctx d_head', normalized_input, Q_W)
V = einsum('samples ctx d_model, d_model d_head -> samples ctx d_head', normalized_input, V_W)
attn_scores = einsum('samples ctx1 d_head, samples ctx2 d_head -> samples ctx1 ctx2', Q, K)
attn_scores = attn_scores.div(np.sqrt(D_HEAD))
attn_scores = apply_causal_mask(attn_scores)
fig = imshow(attn_scores.mean(0))
fig.write_image('attn_scores.png')
attn_weights = attn_scores.softmax(dim=-1)
fig = imshow(attn_weights.mean(0))
fig.write_image('attn_weights.png')
Z = einsum('samples ctx1 d_head, samples ctx1 ctx2 -> samples ctx1 d_head', V, attn_weights)
fig = line(Z.mean(0))
fig.write_image('Z.png')

