"""
Investigate attention to position 0 in BOS@80 experiment.

Checks:
1. Token distribution at position 0 (should be diverse, not constant)
2. Attention to position 0 vs position 80 for all heads
"""

import os
import sys
import numpy as np
import torch
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))

from model_2layer_mechanism import TwoLayerMechanismModel, TwoLayerMechanismConfig

BOS_TOKEN_ID = 50256
BOS_POSITION = 80


def load_model(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = TwoLayerMechanismConfig(
        block_size=128,
        vocab_size=50304,
        n_embd=768,
        n_head=12,
        dropout=0.0,
        norm_type='layernorm',
        use_regression=True,
    )

    model = TwoLayerMechanismModel(model_config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model


def load_data(data_path):
    return np.memmap(data_path, dtype=np.uint16, mode='r')


def get_batch_with_bos(data, batch_size, block_size, bos_position, device):
    tokens_needed = block_size - 1
    ix = np.random.randint(0, len(data) - tokens_needed, size=batch_size)

    sequences = []
    for i in ix:
        before_bos = data[i:i + bos_position].astype(np.int64)
        after_bos = data[i + bos_position:i + tokens_needed].astype(np.int64)
        seq = np.concatenate([before_bos, [BOS_TOKEN_ID], after_bos])
        sequences.append(torch.from_numpy(seq))

    return torch.stack(sequences).to(device)


def main():
    checkpoint_path = '/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/out-2layer-mechanism-bos80/R0/best_ckpt.pt'
    data_path = '/home/nlp/matan_avitan/git/nopos_locating_new/nanoGPT/data/openwebtext/val.bin'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(checkpoint_path, device)
    data = load_data(data_path)

    print("=" * 70)
    print("INVESTIGATION: Attention to Position 0 in BOS@80 Experiment")
    print("=" * 70)

    # Sample multiple batches
    np.random.seed(42)
    n_batches = 10
    batch_size = 64

    all_tokens_pos0 = []
    all_tokens_pos80 = []
    all_attn1 = []
    all_attn2 = []

    for _ in range(n_batches):
        x = get_batch_with_bos(data, batch_size, 128, BOS_POSITION, device)

        # Collect tokens at position 0 and 80
        all_tokens_pos0.extend(x[:, 0].cpu().numpy().tolist())
        all_tokens_pos80.extend(x[:, 80].cpu().numpy().tolist())

        # Get attention
        with torch.no_grad():
            tok_emb = model.wte(x)
            h = model.drop(tok_emb)
            block1_out = model.block1(h, capture_taps=True)
            attn1 = model.block1.attn.last_attention_weights
            block2_out = model.block2(block1_out, capture_taps=True)
            attn2 = model.block2.attn.last_attention_weights

        all_attn1.append(attn1.cpu())
        all_attn2.append(attn2.cpu())

    # Analyze token distribution
    print("\n" + "=" * 70)
    print("1. TOKEN DISTRIBUTION ANALYSIS")
    print("=" * 70)

    pos0_counter = Counter(all_tokens_pos0)
    pos80_counter = Counter(all_tokens_pos80)

    print(f"\nPosition 0: {len(all_tokens_pos0)} samples, {len(pos0_counter)} unique tokens")
    print(f"  - Is BOS (50256) at pos 0? {BOS_TOKEN_ID in pos0_counter}")
    if BOS_TOKEN_ID in pos0_counter:
        print(f"    Count: {pos0_counter[BOS_TOKEN_ID]} ({100*pos0_counter[BOS_TOKEN_ID]/len(all_tokens_pos0):.2f}%)")
    print(f"  - Top 5 tokens at position 0:")
    for token, count in pos0_counter.most_common(5):
        print(f"      Token {token}: {count} times ({100*count/len(all_tokens_pos0):.2f}%)")

    print(f"\nPosition 80: {len(all_tokens_pos80)} samples, {len(pos80_counter)} unique tokens")
    print(f"  - Is BOS (50256) at pos 80? {BOS_TOKEN_ID in pos80_counter}")
    if BOS_TOKEN_ID in pos80_counter:
        print(f"    Count: {pos80_counter[BOS_TOKEN_ID]} ({100*pos80_counter[BOS_TOKEN_ID]/len(all_tokens_pos80):.2f}%)")

    # Analyze attention
    print("\n" + "=" * 70)
    print("2. ATTENTION ANALYSIS")
    print("=" * 70)

    attn1_all = torch.cat(all_attn1, dim=0)  # [N, 12, 128, 128]
    attn2_all = torch.cat(all_attn2, dim=0)

    print("\nBlock 2 - Mean attention to key positions (averaged over all queries that can see them):")
    print("-" * 70)
    print(f"{'Head':<6} {'To Pos 0':<15} {'To Pos 80 (BOS)':<18} {'Ratio (80/0)':<15}")
    print("-" * 70)

    for h in range(12):
        # Attention to position 0: all queries can see it
        attn_to_0 = attn2_all[:, h, :, 0].mean().item()

        # Attention to position 80: only queries >= 80 can see it
        attn_to_80 = attn2_all[:, h, 80:, 80].mean().item()

        ratio = attn_to_80 / attn_to_0 if attn_to_0 > 0.001 else float('inf')

        marker = ""
        if attn_to_0 > 0.1:
            marker += " <-- HIGH TO POS 0"
        if attn_to_80 > 0.1:
            marker += " <-- HIGH TO BOS"

        print(f"{h:<6} {attn_to_0:<15.4f} {attn_to_80:<18.4f} {ratio:<15.2f}{marker}")

    print("-" * 70)

    # Check for position 0 head specifically
    print("\n" + "=" * 70)
    print("3. DETAILED ANALYSIS OF HEADS WITH HIGH ATTENTION TO POSITION 0")
    print("=" * 70)

    for h in range(12):
        attn_to_0 = attn2_all[:, h, :, 0].mean().item()
        if attn_to_0 > 0.05:
            print(f"\nBlock 2 Head {h}:")
            print(f"  Mean attention to position 0: {attn_to_0:.4f}")

            # Breakdown by query position
            print("  Attention to pos 0 by query position (sample):")
            for q in [0, 10, 40, 79, 80, 100, 127]:
                attn_q_to_0 = attn2_all[:, h, q, 0].mean().item()
                print(f"    Query {q:3d} -> Key 0: {attn_q_to_0:.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Check if there's truly a position-0 head
    max_attn_to_0 = max(attn2_all[:, h, :, 0].mean().item() for h in range(12))
    max_attn_to_80 = max(attn2_all[:, h, 80:, 80].mean().item() for h in range(12))

    if max_attn_to_0 > 0.1:
        print(f"\nWARNING: High attention to position 0 detected (max = {max_attn_to_0:.4f})")
        print("This could indicate the model is using position 0 as a reference point")
        print("even though there's no constant token there.")
    else:
        print(f"\nNo significant attention to position 0 (max = {max_attn_to_0:.4f})")

    print(f"Max attention to position 80 (BOS): {max_attn_to_80:.4f}")


if __name__ == '__main__':
    main()
