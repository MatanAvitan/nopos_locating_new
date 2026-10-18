"""
Pretrained Model Analysis
Checks if implicit positional encoding exists in pretrained models like GPT-2.
"""
import torch
import numpy as np
import json
from pathlib import Path
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader, TensorDataset
from utils import device
from scipy.stats import pearsonr
import torch.nn as nn

print("Loading pretrained GPT-2...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
model.eval()

N_CTX = 64
HOOK_NAME = 'blocks.0.ln2.hook_normalized'

print("\n" + "="*60)
print("PRETRAINED MODEL ANALYSIS (GPT-2 Small)")
print("="*60)

# Generate diverse text samples
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test[:1000]')

# Extract activations
print("\nExtracting activations from 1000 samples...")
activations = []
with torch.no_grad():
    for i, text in enumerate(dataset['text'][:1000]):
        if len(text.strip()) < 10:
            continue
        tokens = model.to_tokens(text, prepend_bos=True)
        if tokens.shape[1] < N_CTX:
            continue
        tokens = tokens[:, :N_CTX].to(device)
        
        try:
            _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])
            acts = cache[HOOK_NAME][0].detach().cpu()  # [n_ctx, d_model]
            activations.append(acts)
        except:
            continue
        
        if len(activations) >= 500:
            break
        if (i+1) % 100 == 0:
            print(f"  Processed {i+1} samples, collected {len(activations)} valid")

activations = torch.stack(activations)  # [n_samples, n_ctx, d_model]
print(f"\nCollected {activations.shape[0]} samples")

# Analyze population-level patterns
pop_avg = activations.mean(dim=0)  # [n_ctx, d_model]
pos_pattern = pop_avg.mean(dim=1).numpy()  # [n_ctx]

positions = np.arange(N_CTX)
correlation, p_value = pearsonr(positions, pos_pattern)

print(f"\nPopulation-level position correlation: {correlation:.4f} (p={p_value:.6f})")

# Train MLP probe to predict position
print("\nTraining MLP probe to predict position from LN2 activations...")

class MLPProbe(nn.Module):
    def __init__(self, d_model, n_ctx):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_ctx)
        )
    
    def forward(self, x):
        return self.net(x)

d_model = activations.shape[2]
probe = MLPProbe(d_model, N_CTX).to(device)
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

# Prepare data
X = activations.reshape(-1, d_model)  # [n_samples * n_ctx, d_model]
y = torch.arange(N_CTX).repeat(activations.shape[0])  # [n_samples * n_ctx]

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=2048, shuffle=True)

# Train probe
probe.train()
for epoch in range(50):
    total_loss = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        logits = probe(batch_x)
        loss = nn.CrossEntropyLoss()(logits, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

# Evaluate probe
probe.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = probe(batch_x)
        preds = logits.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

probe_accuracy = correct / total
print(f"\nMLP Probe Accuracy: {probe_accuracy:.4f}")

# Save results
output_file = Path('results/pretrained_analysis_results.json')
output_file.parent.mkdir(parents=True, exist_ok=True)

results = {
    'model': 'gpt2-small',
    'n_samples': int(activations.shape[0]),
    'implicit_position_correlation': float(correlation),
    'correlation_p_value': float(p_value),
    'probe_accuracy': float(probe_accuracy),
    'population_pattern_strength': float(pos_pattern.var()),
}

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Implicit position signal: {correlation:.4f}")
print(f"Probe accuracy: {probe_accuracy:.4f}")
print("="*60)
