"""
Quick comparison of attention patterns between trained and random NoPE models.

Key question: Does training make attention patterns non-uniform?
If so, this could explain why the 1/(i+1) variance mechanism breaks down.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nanoGPT"))
from model_nope import GPT, GPTConfig


def load_trained_model(checkpoint_path, device="cuda"):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint["model_args"]
    config = GPTConfig(**model_args)
    config.log_attention_stats = True  # Enable to capture attention weights
    model = GPT(config)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


def create_random_model(config, device="cuda"):
    """Create randomly initialized model with same config."""
    config.log_attention_stats = True
    model = GPT(config)
    model.to(device)
    model.eval()
    return model


def get_attention_weights(model, input_ids, device="cuda"):
    """Extract attention weights from model by manually computing attention."""
    import math

    model.eval()
    with torch.no_grad():
        # Get embeddings
        tok_emb = model.transformer.wte(input_ids)
        x = model.transformer.drop(tok_emb)
        block = model.transformer.h[0]

        # Apply LN1
        x_ln1 = block.ln_1(x)

        # Get Q, K, V
        B, T, C = x_ln1.shape
        n_head = block.attn.n_head

        q, k, v = block.attn.c_attn(x_ln1).split(block.attn.n_embd, dim=2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)

        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        att = att.masked_fill(causal_mask, float("-inf"))
        att = torch.nn.functional.softmax(att, dim=-1)

        return att.cpu()


def compute_attention_entropy(attn_weights):
    """Compute entropy of attention distribution for each query position."""
    # attn_weights: [batch, n_heads, seq, seq]
    # For causal attention, only lower triangle is valid

    batch, n_heads, seq_len, _ = attn_weights.shape

    # Compute entropy per query position (averaged over batch and heads)
    entropies = []
    for pos in range(seq_len):
        # Get attention distribution for this query position
        # Only first pos+1 values are non-zero due to causal mask
        attn_slice = attn_weights[:, :, pos, : pos + 1]  # [batch, n_heads, pos+1]

        # Compute entropy
        eps = 1e-10
        ent = -torch.sum(
            attn_slice * torch.log(attn_slice + eps), dim=-1
        )  # [batch, n_heads]
        entropies.append(ent.mean().item())

    return entropies


def compute_attention_uniformity(attn_weights):
    """
    Compute how uniform the attention is compared to theoretical uniform attention.

    For uniform causal attention: each position attends equally to all previous tokens
    Theoretical entropy at position i = log(i+1)
    """
    batch, n_heads, seq_len, _ = attn_weights.shape

    uniformity_scores = []
    for pos in range(seq_len):
        # Theoretical entropy for uniform attention over pos+1 tokens
        theoretical_entropy = np.log(pos + 1)

        # Actual entropy
        attn_slice = attn_weights[:, :, pos, : pos + 1]
        eps = 1e-10
        actual_entropy = (
            -torch.sum(attn_slice * torch.log(attn_slice + eps), dim=-1).mean().item()
        )

        # Uniformity = actual / theoretical (1 = perfectly uniform)
        if theoretical_entropy > 0:
            uniformity = actual_entropy / theoretical_entropy
        else:
            uniformity = 1.0
        uniformity_scores.append(uniformity)

    return uniformity_scores


def compute_first_token_attention(attn_weights):
    """
    Compute how much attention goes to the first token (position 0).

    In many trained models, attention to first token is disproportionately high.
    """
    batch, n_heads, seq_len, _ = attn_weights.shape

    first_token_attention = []
    for pos in range(seq_len):
        # Attention to first token from position pos
        attn_to_first = attn_weights[:, :, pos, 0].mean().item()
        first_token_attention.append(attn_to_first)

    return first_token_attention


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default="nanoGPT/out-nope-1layer-ln/ckpt.pt"
    )
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--output_dir", type=str, default="results/attention_comparison"
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ATTENTION PATTERN COMPARISON: TRAINED VS RANDOM")
    print("=" * 70)

    # Load trained model
    print(f"\n1. Loading trained model from {args.checkpoint}")
    trained_model, config = load_trained_model(args.checkpoint, args.device)

    # Create random model with same config
    print("2. Creating randomly initialized model")
    random_model = create_random_model(config, args.device)

    seq_len = config.block_size
    vocab_size = config.vocab_size

    print(f"\nConfig: seq_len={seq_len}, vocab_size={vocab_size}")

    # Generate random sequences
    print(f"\n3. Generating {args.n_samples} random sequences...")
    torch.manual_seed(42)
    input_ids = torch.randint(
        0, vocab_size, (args.n_samples, seq_len), device=args.device
    )

    # Process in batches
    batch_size = 20
    trained_entropies = []
    trained_uniformity = []
    trained_first_token = []
    random_entropies = []
    random_uniformity = []
    random_first_token = []

    print("4. Computing attention statistics...")
    for i in range(0, args.n_samples, batch_size):
        batch = input_ids[i : i + batch_size]

        # Trained model
        trained_attn = get_attention_weights(trained_model, batch, args.device)
        if trained_attn is not None:
            trained_entropies.append(compute_attention_entropy(trained_attn))
            trained_uniformity.append(compute_attention_uniformity(trained_attn))
            trained_first_token.append(compute_first_token_attention(trained_attn))

        # Random model
        random_attn = get_attention_weights(random_model, batch, args.device)
        if random_attn is not None:
            random_entropies.append(compute_attention_entropy(random_attn))
            random_uniformity.append(compute_attention_uniformity(random_attn))
            random_first_token.append(compute_first_token_attention(random_attn))

    if not trained_entropies or not random_entropies:
        print(
            "\nERROR: Could not capture attention weights. Check model configuration."
        )
        return

    # Average across batches
    trained_entropies = np.mean(trained_entropies, axis=0)
    trained_uniformity = np.mean(trained_uniformity, axis=0)
    trained_first_token = np.mean(trained_first_token, axis=0)
    random_entropies = np.mean(random_entropies, axis=0)
    random_uniformity = np.mean(random_uniformity, axis=0)
    random_first_token = np.mean(random_first_token, axis=0)

    # Compute correlations
    positions = np.arange(seq_len)
    theoretical_entropy = np.log(positions + 1)

    r_trained_theory, _ = pearsonr(trained_entropies, theoretical_entropy)
    r_random_theory, _ = pearsonr(random_entropies, theoretical_entropy)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"""
ENTROPY VS THEORETICAL (log(i+1)):
  Trained: r = {r_trained_theory:.4f}
  Random:  r = {r_random_theory:.4f}
  
UNIFORMITY (actual_entropy / theoretical_entropy):
  Trained avg: {np.mean(trained_uniformity):.4f}
  Random avg:  {np.mean(random_uniformity):.4f}
  
ATTENTION TO FIRST TOKEN:
  Trained at pos 10:  {trained_first_token[10]:.4f} (expected: {1 / 11:.4f} for uniform)
  Random at pos 10:   {random_first_token[10]:.4f}
  Trained at pos 100: {trained_first_token[100]:.4f} (expected: {1 / 101:.4f} for uniform)
  Random at pos 100:  {random_first_token[100]:.4f}
  
DEVIATION FROM UNIFORM:
  Trained first-token attention decay: {trained_first_token[1]:.4f} -> {trained_first_token[-1]:.4f}
  Random first-token attention decay:  {random_first_token[1]:.4f} -> {random_first_token[-1]:.4f}
""")

    # Key insight
    trained_is_uniform = np.mean(trained_uniformity) > 0.95
    random_is_uniform = np.mean(random_uniformity) > 0.95

    print("=" * 70)
    print("KEY FINDING")
    print("=" * 70)

    if trained_is_uniform and random_is_uniform:
        print("""
Both trained and random models have approximately UNIFORM attention.
This means the 1/(i+1) variance mechanism should work in both!
The difference must be elsewhere (MLP? token statistics?).
""")
    elif random_is_uniform and not trained_is_uniform:
        print("""
TRAINED model has NON-UNIFORM attention while random is uniform!
This explains why the 1/(i+1) variance mechanism breaks:
- Training learns to attend non-uniformly
- This breaks the variance decay that encodes position
""")
    else:
        print("""
Both models show some non-uniformity.
Need to investigate further what's happening.
""")

    # Save results
    results = {
        "trained": {
            "entropy_by_position": trained_entropies.tolist(),
            "uniformity_by_position": trained_uniformity.tolist(),
            "first_token_attention": trained_first_token.tolist(),
            "entropy_correlation_with_theory": r_trained_theory,
            "mean_uniformity": float(np.mean(trained_uniformity)),
        },
        "random": {
            "entropy_by_position": random_entropies.tolist(),
            "uniformity_by_position": random_uniformity.tolist(),
            "first_token_attention": random_first_token.tolist(),
            "entropy_correlation_with_theory": r_random_theory,
            "mean_uniformity": float(np.mean(random_uniformity)),
        },
    }

    results_path = Path(args.output_dir) / "attention_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
