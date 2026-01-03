import glob
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformer_lens.utils import to_numpy
import plotly.express as px

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Utils
def line(tensor, line_labels=None, yaxis="", xaxis="", title="", legend_title="", save_path=None, **kwargs):
    tensor = to_numpy(tensor)
    # Use Plotly's white template and enforce a clean layout
    fig = px.line(tensor, template="plotly_white", **kwargs)

    # Update layout with enhanced aesthetics
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, family="Serif")),
        xaxis=dict(title=xaxis, title_font=dict(size=18), tickfont=dict(size=16)),
        yaxis=dict(title=yaxis, title_font=dict(size=18), tickfont=dict(size=16)),
        legend=dict(title=legend_title, font=dict(size=16)),
        width=800,
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # Increase default line width and set marker styles
    for trace in fig.data:
        trace.line.width = 3
        trace.marker = dict(symbol="circle", size=8)

    # Apply line labels if provided
    if line_labels:
        for c, label in enumerate(line_labels):
            fig.data[c].name = label

    # Save or show
    if save_path:
        fig.write_image(f"{save_path}.png", width=800, height=500, scale=2)  # 300 DPI
        fig.write_image(f"{save_path}.pdf")
        print(f"Saved figure to {save_path}.png and {save_path}.pdf")
    else:
        fig.show()

    return fig


def imshow(tensor, yaxis="", xaxis="", save_path=None, **kwargs):
    tensor = to_numpy(tensor)
    # Use a high-quality continuous color scale and a white template
    plot_kwargs = {
        "color_continuous_scale": "RdBu_r",
        "color_continuous_midpoint": 0.0,
        "labels": {"x": xaxis, "y": yaxis},
        "template": "plotly_white",
        "aspect": "equal"
    }
    if 'title' in kwargs and kwargs['title']:
        title = kwargs['title']
    else:
        title = 'Image'
    plot_kwargs.update(kwargs)
    fig = px.imshow(tensor, **plot_kwargs)
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, family="Serif")),
        xaxis=dict(title=xaxis, title_font=dict(size=18), tickfont=dict(size=16)),
        yaxis=dict(title=yaxis, title_font=dict(size=18), tickfont=dict(size=16)),
        width=800,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # Save or show
    if save_path:
        fig.write_image(f"{save_path}.png", width=800, height=600, scale=2)  # 300 DPI
        fig.write_image(f"{save_path}.pdf")
        print(f"Saved figure to {save_path}.png and {save_path}.pdf")
    else:
        fig.show()

    return fig


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

def tokenize_fn(examples, tokenizer, max_length):
    return tokenizer(
        examples,
        truncation=True,
        padding='max_length',
        max_length=max_length
    )

def store_acts(model, loader, layers_to_cache, cache_acts_dir):
    model.eval()
    with torch.no_grad():
        for i, (tokens, _) in enumerate(tqdm(loader, desc="Caching loader")):
            tokens = tokens.to(device, non_blocking=True)
            try:
                _, cache = model.run_with_cache(tokens, names_filter=layers_to_cache)
                act = cache['blocks.0.ln2.hook_normalized'].detach().cpu()  # [B, N_CTX, D_MODEL]

                # Save to disk batch-wise
                torch.save(act, f"{cache_acts_dir}/batch_{i:05d}.pt")

                del cache, act
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"Skipping batch {i} due to OOM.")
                    torch.cuda.empty_cache()
                else:
                    raise

class CachedActivationDataset(Dataset):
        def __init__(self, file_paths):
            self.file_paths = file_paths
            self.index_map = []
            for file_idx, path in enumerate(file_paths):
                acts = torch.load(path, map_location='cpu')
                B, N, _ = acts.shape
                for i in range(B * N):
                    self.index_map.append((file_idx, i))
            self.seq_len = N

        def __len__(self):
            return len(self.index_map)

        def __getitem__(self, idx):
            file_idx, flat_idx = self.index_map[idx]
            act = torch.load(self.file_paths[file_idx], map_location='cpu')  # [B, N, D]
            B, N, D = act.shape
            b_idx = flat_idx // N
            pos = flat_idx % N
            return act[b_idx, pos], pos  # [D_MODEL], int

def load_acts_loader(cache_acts_dir):
    # ——— Reload cached activations for training ———
    acts_files = sorted(glob.glob(f"{cache_acts_dir}/batch_*.pt"))
    cached_ds = CachedActivationDataset(acts_files)
    cached_loader = DataLoader(
        cached_ds, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=True, num_workers=16, prefetch_factor=2
    )
    return cached_loader


